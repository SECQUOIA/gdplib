# Vinyl Production 
Information comes from methanol model and will be edited later
This is a GDP model to determine the optimal maximum profit for a methanol production process.

## infeasible NLP model 
import pyomo.environ as pe
import pyomo.gdp as gdp
import pandas as pd 
from pyomo.util.infeasible import log_infeasible_constraints
from pyomo.util.model_size import build_model_size_report
from pyomo.contrib.fbbt.fbbt import fbbt, compute_bounds_on_expr

class VinylChlorideModel(object):
    def __init__(self):
        self.model = m =  pe.ConcreteModel() # main model
        ## parameters
        self.alpha = 0.3665 #compressor coefficient
        self.compeff = 0.750 #compressor efficiency
        self.gamma = 1.3 #ratio of Cp to Cv
        self.disteff = 0.5 #distillation column tray efficiency
        self.eps1 = 1e-4
        ## Sets define 
        m.stream = pe.RangeSet(71)
        m.compressor = pe.RangeSet(8)
        m.distillation_column = pe.RangeSet(5)
        m.flash = pe.RangeSet(1)
        m.pyrolysis_furnance = pe.RangeSet(2)
        m.cooler = pe.RangeSet(5)
        m.heater = pe.RangeSet(5)
        m.pump = pe.RangeSet(2)
        m.valve = pe.RangeSet(1)
        m.feed = pe.RangeSet(2)
        m.direct_chlorination_reactor = pe.RangeSet(1)
        m.oxychlorination_reactor = pe.RangeSet(1)
        m.components = pe.Set(initialize = ['nitrogen','oxygen','ethylene','hydrogen chloride','vinyl chloride','chlorine','ethylene di-chloride','water'])

        # parameter related to costs (in unites of $1000)
        m.costelec = pe.Param(initialize = 0.340,doc ="electricity cost,unit kw-year")
        m.costqc = pe.Param(initialize = 0.70,doc ="cooling cost,unit: 1e9KJ")
        m.costqh = pe.Param(initialize = 8.0,doc = "heating cost,unit: 1e9KJ")
        m.costfuel = pe.Param(initialize = 4.0,doc = "fuel cost furnace,unit: 1e9KJ")
        m.catalyst_replacement = pe.Param(initialize = 1.25,doc = "catalyst replacement in a year")
        m.material_of_construction_factor = pe.Param(initialize = 1.5,doc = "material of construction factor")
        m.pressure_factor = pe.Param(initialize = 1.7,doc = "pressure factor")
        m.residence_time = pe.Param(initialize = 5.0,doc = "residence time in flash,unit:min")
        m.marshall_swift_index = pe.Param(initialize = 1031.8,doc = "marshall and swift index")
        #Variables 
        COOLER_COST = dict()
        COOLER_COST[1] = 8.88311
        COOLER_COST[2] = 0
        COOLER_COST[3] = 7.50749
        COOLER_COST[4] = 12.2928
        COOLER_COST[5] = 2.90323
        m.cooler_cost = pe.Var(m.cooler,within = pe.NonNegativeReals,doc= 'cost for each cooler',initialize = COOLER_COST)
        m.utility_requirement_c = pe.Var(m.cooler,domain = pe.NonNegativeReals,doc = 'unit: 1e12 Kj/yr, for cooler',initialize = 0)
        COMPRESSOR_COST = dict()
        COMPRESSOR_COST[1] = 1586.23
        COMPRESSOR_COST[2] = 0
        COMPRESSOR_COST[3] = 0
        COMPRESSOR_COST[4] = 5.10317
        COMPRESSOR_COST[5] = 0
        COMPRESSOR_COST[6] = 5202.15
        COMPRESSOR_COST[7] = 1738.81
        COMPRESSOR_COST[8] = 5.10317
        m.compressor_cost = pe.Var(m.compressor,within = pe.NonNegativeReals,doc= 'cost for each compressor',initialize = COMPRESSOR_COST)
        m.electricity_requirement = pe.Var(m.compressor,domain = pe.NonNegativeReals,doc = 'unit:kW',initialize = 0) 
        m.furnace_cost = pe.Var(m.pyrolysis_furnance,within = pe.NonNegativeReals,doc = 'cost for each furnace',initialize = 379.413)
        m.cost_of_distillation = pe.Var(m.distillation_column,within = pe.NonNegativeReals,doc = 'cost for each distillation',bounds = (0,1e6),initialize = 1)
        HEATER_COST = dict()
        HEATER_COST[1] = 4.38869
        HEATER_COST[2] = 3.46175
        HEATER_COST[3] = 2.90323
        HEATER_COST[4] = 10.899
        HEATER_COST[5] = 4.95907
        m.heater_cost = pe.Var(m.heater,within = pe.PositiveReals,doc= 'cost for each heater',initialize = HEATER_COST,bounds = (0,1e7))
        m.utility_requirement_h = pe.Var(m.heater,domain = pe.Reals,doc = 'unit: 1e12 Kj/yr',initialize = 0, bounds = (0,1e7))
        m.cost_of_oxychlorination_reactor = pe.Var(m.oxychlorination_reactor,within = pe.PositiveReals,initialize = 1,bounds = (0,1e7))
        m.cost_of_direct_chlorination_reactor = pe.Var(m.direct_chlorination_reactor,within = pe.PositiveReals,initialize = 1,bounds = (0,1e7))
        m.cost_of_flash = pe.Var(m.flash,within = pe.PositiveReals,initialize = 1,bounds = (0,1e7))

        ##stream matches


        self.equipments = ['compressor','cooler','heater','furnace','distillation','direct','oxychlorination','pump','valve','flash']
        self.inlet_streams  = dict()
        self.outlet_streams = dict()
        self.vapor_outlets = dict()
        self.liquid_outlets = dict()
        self.light_key_components = dict()
        self.heavy_key_components = dict()
        self.light_components = dict()
        self.heavy_components = dict()
        self.key_components = dict()
        self.flash_key_components = dict()
        for i in self.equipments:
            self.inlet_streams[i] = dict()
            self.outlet_streams[i] = dict()
        self.vapor_outlets['flash'] = dict()
        self.vapor_outlets['distillation'] = dict()
        self.liquid_outlets['flash'] = dict()
        self.liquid_outlets['distillation'] = dict()
        self.light_key_components['distillation'] = dict()
        self.light_components['distillation'] = dict()
        self.heavy_key_components['distillation'] = dict()
        self.heavy_components['distillation'] = dict()
        self.key_components['distillation'] = dict()
        self.flash_key_components['flash'] = dict()
        #inlet streams 
        self.inlet_streams['compressor'][1] = 12
        self.inlet_streams['compressor'][2] = 13
        self.inlet_streams['cooler'][2] = 15
        self.inlet_streams['compressor'][3] = 16 
        self.inlet_streams['compressor'][4] = 27 
        self.inlet_streams['heater'][2] = 18
        self.inlet_streams['heater'][3] = 20
        self.inlet_streams['oxychlorination'][1] = 22
        self.inlet_streams['cooler'][3] = 23
        self.inlet_streams['flash'][1] = 24
        self.inlet_streams['distillation'][1] = 26
        self.inlet_streams['heater'][1] = 5
        self.inlet_streams['direct'][1] = 8
        self.inlet_streams['cooler'][1] = 9
        self.inlet_streams['compressor'][5] = 33
        self.inlet_streams['compressor'][6] = 34
        self.inlet_streams['cooler'][4] = 36 
        self.inlet_streams['compressor'][7] = 37
        self.inlet_streams['compressor'][8] = 66
        self.inlet_streams['furnace'][1] = 41
        self.inlet_streams['furnace'][2] = 42
        self.inlet_streams['cooler'][5] = 45
        self.inlet_streams['pump'][1] = 46
        self.inlet_streams['pump'][2] = 63
        self.inlet_streams['valve'][1] = 47
        self.inlet_streams['distillation'][2] = 49
        self.inlet_streams['distillation'][3] = 50
        self.inlet_streams['distillation'][4] = 51
        self.inlet_streams['distillation'][5] = 58
        self.inlet_streams['heater'][4] = 64
        self.inlet_streams['heater'][5] = 67

        # outlet streams 
        self.outlet_streams['compressor'][1] = 14
        self.outlet_streams['compressor'][2] = 15
        self.outlet_streams['cooler'][1] = 10
        self.outlet_streams['cooler'][2] = 16
        self.outlet_streams['compressor'][3] = 17 
        self.outlet_streams['compressor'][4] = 29
        self.outlet_streams['heater'][1] = 7
        self.outlet_streams['heater'][2] = 19
        self.outlet_streams['heater'][3] = 21
        self.outlet_streams['heater'][4] = 65
        self.outlet_streams['heater'][5] = 68
        self.outlet_streams['oxychlorination'][1] = 23
        self.outlet_streams['cooler'][3] = 24
        self.outlet_streams['direct'][1] = 9
        self.outlet_streams['cooler'][1] = 10
        self.outlet_streams['compressor'][5] = 35
        self.outlet_streams['compressor'][6] = 36
        self.outlet_streams['cooler'][4] = 37 
        self.outlet_streams['compressor'][7] = 38
        self.outlet_streams['compressor'][8] = 67
        self.outlet_streams['furnace'][1] = 43
        self.outlet_streams['furnace'][2] = 44
        self.outlet_streams['cooler'][5] = 46
        self.outlet_streams['pump'][1] = 47
        self.outlet_streams['pump'][2] = 64
        self.outlet_streams['valve'][1] = 48
        #vapor outlets
        self.vapor_outlets['flash'][1] = 25
        self.vapor_outlets['distillation'][1] = 30
        self.vapor_outlets['distillation'][2] = 51
        self.vapor_outlets['distillation'][3] = 57
        self.vapor_outlets['distillation'][4] = 53
        self.vapor_outlets['distillation'][5] = 61
        #liquid outlets 
        self.liquid_outlets['flash'][1] = 26
        self.liquid_outlets['distillation'][1] = 31
        self.liquid_outlets['distillation'][2] = 52
        self.liquid_outlets['distillation'][3] = 58
        self.liquid_outlets['distillation'][4] = 54
        self.liquid_outlets['distillation'][5] = 62
        # light key component
        self.light_key_components['distillation'][1] = 'ethylene di-chloride'
        self.light_key_components['distillation'][2] = 'vinyl chloride'
        self.light_key_components['distillation'][3] = 'hydrogen chloride'
        self.light_key_components['distillation'][4] = 'hydrogen chloride'
        self.light_key_components['distillation'][5] = 'vinyl chloride'
        # light component
        self.light_components['distillation'][1] = 'ethylene di-chloride'
        self.light_components['distillation'][2] = 'hydrogen chloride'
        self.light_components['distillation'][3] = 'hydrogen chloride'
        self.light_components['distillation'][4] = 'hydrogen chloride'
        self.light_components['distillation'][5]  = 'vinyl chloride'
        # heavy key component 
        self.heavy_key_components['distillation'][1]  = 'water'
        self.heavy_key_components['distillation'][2]  = 'ethylene di-chloride'
        self.heavy_key_components['distillation'][3]  = 'vinyl chloride'
        self.heavy_key_components['distillation'][4]  = 'vinyl chloride'
        self.heavy_key_components['distillation'][5] = 'ethylene di-chloride'
        # heavy component 
        self.heavy_components['distillation'][1]  = 'water'
        self.heavy_components['distillation'][2]  = 'ethylene di-chloride'
        self.heavy_components['distillation'][3]  = 'ethylene di-chloride'
        self.heavy_components['distillation'][4]  = 'vinyl chloride'
        self.heavy_components['distillation'][5]  = 'ethylene di-chloride'
        # key component 
        for unit_number in range(1,6):
            self.key_components['distillation'][unit_number] = [self.heavy_key_components['distillation'][unit_number] ,self.light_key_components['distillation'][unit_number]]
        #flash key component
        self.flash_key_components['flash'][1] = 'hydrogen chloride'

        ## parameters with index
        a = {}
        a['vinyl chloride'] = 20750
        m.heat_vapor = pe.Param(m.components,initialize= a ,default = 0)
        b = {}
        b[1] = -182213
        m.heat_of_reaction_d = pe.Param(m.direct_chlorination_reactor,initialize = b )
        c = {}
        c[1] = -239450
        m.heat_of_reaction_o = pe.Param(m.oxychlorination_reactor,initialize = c)
        d = {}
        d[1] = 72760
        d[2] = 72760
        m.heat_of_reaction_p = pe.Param(m.pyrolysis_furnance,initialize = d)
        M = {}
        M[1] = 5.0
        M[2] = 30.0 
        m.upper_bound_pressure = pe.Param(m.pyrolysis_furnance,initialize = M)
        n = {}
        n[1] = 15.90
        n[2] = 17.80
        m.upper_bound_consumption = pe.Param(m.pyrolysis_furnance,initialize = n)
        e = {}
        e[1] = 1.2
        e[2] = 1.5
        m.material_of_construction_factor_f = pe.Param(m.pyrolysis_furnance,initialize = e)
        f = {}
        f[1] = 12
        f[2] = 950
        f[3] = 50
        f[4] = 36
        f[5] = 45
        m.average_flow = pe.Param(m.distillation_column,initialize = f)
        g = {}
        g['ethylene'] = 1.
        m.stream_1_composition = pe.Param(m.components,initialize = g,default = 0)
        h = {}
        h['chlorine']  = 1.
        m.stream_2_composition = pe.Param(m.components,initialize = h,default = 0)
        i = {}
        i['nitrogen'] = 0.79
        i['oxygen'] = 0.21
        m.stream_3_composition = pe.Param(m.components,initialize = i,default = 0)
        j = {}
        j['oxygen'] = 1.0
        m.stream_4_composition = pe.Param(m.components,initialize = j,default = 0)
        self.CPpure = {}
        self.CPpure['nitrogen'] = 25
        self.CPpure['oxygen'] = 25
        self.CPpure['ethylene'] = 50
        self.CPpure['hydrogen chloride'] = 40
        self.CPpure['vinyl chloride'] = 200
        self.CPpure['chlorine'] = 100
        self.CPpure['ethylene di-chloride'] = 400
        self.CPpure['water'] = 100
        k = {}
        k[41] = 0.0006
        k[42] = 0.00072
        m.conversion_corelation = pe.Param(m.stream,initialize = k, default = 0)

        ##parameter set with pandas
        self.Antoine = pd.read_csv('antoine_coefficient_1.csv')
        self.antoine = self.Antoine.set_index('name').T.to_dict('list')
        self.gcomp = pd.read_csv('Gcomp.csv')
        self.gcomp = self.gcomp.fillna(0)
        self.Gcomp = self.gcomp.set_index('stream').to_dict('list')
        self.Stream_CP = list()
        for stream in m.stream:
            self.Stream_CP.append(sum(self.Gcomp[comp][stream-1] * self.CPpure[comp] for comp in m.components))
        ## Variables     
        m.split_fraction = pe.Var(m.stream, domain = pe.NonNegativeReals,bounds = (0,1),initialize = 0)
        m.split_fraction[27].setub(0)
        m.stream_flowrate = pe.Var(m.stream,domain = pe.NonNegativeReals,bounds = (0,50),initialize = 6)
        m.component_stream_flowrate = pe.Var(m.stream,m.components,domain = pe.NonNegativeReals,bounds = (0,50))
        m.pressure = pe.Var(m.stream,domain = pe.NonNegativeReals,bounds = (0,30),initialize = 2.0) 
        ## bounds
        m.pressure[8].setlb(1)
        m.pressure[8].setub(2)
        m.temperature = pe.Var(m.stream,domain = pe.NonNegativeReals,bounds = (None,10)) 
        m.vapor_pressure = pe.Var(m.stream,m.components,domain = pe.Reals,bounds = (0,400),initialize = 5) # ?? doc for unit for pe.Variables
        m.temperature[1].fix(3)
        m.temperature[8].setlb(3.23)
        m.temperature[8].setub(3.38)
        for stream in [21,22]:
            m.temperature[stream].setlb(4.93)
            m.temperature[stream].setub(5.08)
        for stream in [10,12,36,37,38,45,46]:
            m.temperature[stream].setlb(3.0)
        for stream in [30,31]:
            m.temperature[stream].setlb(2.5)
            m.temperature[stream].setub(7.5)
        for stream in [24,25,26]:
            m.temperature[stream].setlb(2.5)
            m.temperature[stream].setub(5.5)
        for stream in [36,37,38,42,44]:
            m.temperature[stream].setub(8.23)
        m.temperature[41].setub(9.5)
        m.temperature[42].setlb(7.73)
        m.temperature[42].setub(8.23)
        m.temperature[44].setlb(7.73)
        m.temperature[44].setub(8.23)
        m.temperature[45].setlb(3.0)
        m.temperature[45].setub(7.8)
        m.temperature[46].setlb(3.0)
        m.temperature[46].setub(7.8)
        for stream in [50,57,58,61,62]:
            m.temperature[stream].setlb(3.0)
            m.temperature[stream].setub(9.5)
        m.temperature[68].setlb(4.93)
        m.temperature[68].setub(5.08)

        for stream in [12,27,29,30,31,50,57,61,62,66,68]:
            m.pressure[stream].setlb(1.0)
        for stream in [21,22,24,25,26,67]:
            m.pressure[stream].setlb(2.5)
            m.pressure[stream].setub(6.5)
        m.pressure[68].setub(6.5)
        m.pressure[30].setub(7.5)
        m.pressure[31].setub(7.5)
        m.pressure[42].setlb(25)
        m.pressure[44].setlb(25)
        m.pressure[36].setlb(5)
        m.pressure[36].setub(15)
        m.pressure[37].setlb(5)
        m.pressure[37].setub(15)
        m.pressure[38].setlb(5)

        for stream in [1,2,5,6,7,9,10,20,21,23,24,32,34,70]:
            m.stream_flowrate[stream].setlb(1.0)
        self.bounds_= pd.read_csv('VP_bounds.csv')
        self.stream_number = self.bounds_.iloc[:,[0]].dropna().values
        self.comp_number = self.bounds_.iloc[:,[1]].dropna().values
        self.VP_lb = self.bounds_.iloc[:,[2]].dropna().values
        self.VP_ub = self.bounds_.iloc[:,[3]].dropna().values
        for i in range(len(self.stream_number)):
            m.vapor_pressure[self.stream_number[i,0],self.comp_number[i,0]].setlb(self.VP_lb[i,0])
            m.vapor_pressure[self.stream_number[i,0],self.comp_number[i,0]].setub(self.VP_ub[i,0])
        #initialization
        m.split_fraction[5] = 0.5
        m.split_fraction[59] = 0.8
        m.electricity_requirement[1] = 20.7362
        m.electricity_requirement[6] = 111.779
        m.electricity_requirement[7] = 23.6241
        m.utility_requirement_c[1] = 1.27335
        m.utility_requirement_c[3] = 0.31595
        m.utility_requirement_c[4] = 0.646112
        m.utility_requirement_h[1] = 0.00522984
        m.utility_requirement_h[2] = 0.00348001
        m.utility_requirement_h[4] = 1.01767
        m.utility_requirement_h[5] = 0.0702163
        m.slack_variable_1_P = pe.Var(domain = pe.NonNegativeReals,initialize =  0)
        m.slack_variable_1_N = pe.Var(domain = pe.NonNegativeReals,initialize =  0)
        m.slack_variable_2_P = pe.Var(domain = pe.NonNegativeReals,initialize =  0)
        m.slack_variable_2_N = pe.Var(domain = pe.NonNegativeReals,initialize =  0)
        m.slack_variable_3_P = pe.Var(domain = pe.NonNegativeReals,initialize =  0)
        m.slack_variable_3_N = pe.Var(domain = pe.NonNegativeReals,initialize =  0)
        ## initialization with pandas
        self.ini_ = pd.read_csv('bounds.csv')
        self.stream_number = self.ini_.iloc[:,[0]].dropna().values
        self.comp_number = self.ini_.iloc[:,[4]].dropna().values
        self.comp_comp = self.ini_.iloc[:,[5]].dropna().values
        self.vp_number = self.ini_.iloc[:,[7]].dropna().values
        self.vp_comp = self.ini_.iloc[:,[8]].dropna().values
        

        for i in range(len(self.stream_number)):
            m.stream_flowrate[self.stream_number[i,0]] = self.ini_.iloc[:,[1]].dropna().values[i,0]
            m.pressure[self.stream_number[i,0]] = self.ini_.iloc[:,[2]].dropna().values[i,0]
            m.temperature[self.stream_number[i,0]] = self.ini_.iloc[:,[3]].dropna().values[i,0]
        for i in range(len(self.comp_number)):
            m.component_stream_flowrate[self.comp_number[i,0],self.comp_comp[i,0]] = self.ini_.iloc[:,[6]].dropna().values[i,0]
        for i in range(len(self.vp_number)):
            m.vapor_pressure[self.vp_number[i,0],self.vp_comp[i,0]] = self.ini_.iloc[:,[9]].dropna().values[i,0]
        #***********************
        ##global constraints
        #***********************
        m.stream_flowrate[1].fix(17.8341)
        m.stream_flowrate[2].fix(8.91704)
        m.stream_flowrate[4].fix(4.45852)
        def component_match(_m,_c):
            return m.stream_flowrate[_c] == sum(m.component_stream_flowrate[_c,d] for d in m.components)
        m.Component_match = pe.Constraint(m.stream,rule = component_match)
        def production_specify(_m):
            return m.component_stream_flowrate[70,'vinyl chloride'] >= 0.90 * m.stream_flowrate[70]
        m.Production_specify = pe.Constraint(rule = production_specify)
        def ethylene_inlet_stream_specify(_m,_c):
            return m.component_stream_flowrate[1,_c] == m.stream_flowrate[1] * m.stream_1_composition[_c]
        m.Ethylene_inlet_stream_specify = pe.Constraint(m.components, rule = ethylene_inlet_stream_specify)
        def chlorine_inlet_stream_specify(_m,_c):
            return m.component_stream_flowrate[2,_c] == m.stream_flowrate[2] * m.stream_2_composition[_c]
        m.chlorine_inlet_stream_specify = pe.Constraint(m.components,rule = chlorine_inlet_stream_specify)
        def oxygen_inlet_stream_specify(_m_,_c):
            return m.component_stream_flowrate[4,_c] == m.stream_flowrate[4] * m.stream_4_composition[_c]
        m.Oxygen_inlet_stream_specify = pe.Constraint(m.components,rule = oxygen_inlet_stream_specify)
        def air_inlet_stream_specify(_m,_c):
            return m.component_stream_flowrate[3,_c] == m.stream_flowrate[3] * m.stream_3_composition[_c]
        m.Air_inlet_stream_specify = pe.Constraint(m.components,rule = air_inlet_stream_specify)
        def stream_join_flowsheet(_m,_c):
            return m.component_stream_flowrate[40,_c] == m.component_stream_flowrate[39,_c] + m.component_stream_flowrate[65,_c] 
        m.Stream_join_flowsheet = pe.Constraint(m.components,rule = stream_join_flowsheet)
        def temp_join_flowsheet(_m):
            return m.temperature[40] == m.temperature[65]
        m.Temp_join_flowsheet_2 = pe.Constraint(rule = temp_join_flowsheet)
        def p_join_flowsheet_2(_m):
            return m.pressure[40] == m.pressure[65]
        m.P_join_flowsheet_2 = pe.Constraint(rule = p_join_flowsheet_2)
        def p_join_flowsheet_1(_m):
            return m.pressure[40] == m.pressure[39]
        m.P_join_flowsheet_1 = pe.Constraint(rule = p_join_flowsheet_1)
        # ************************************************
        # compressor selection ---- outside disjunction 
        # ************************************************

        self.build_stream_doesnt_exist(m,33)
        self.build_stream_doesnt_exist(m,35)
        self.build_equal_streams(m,32,34)
        self.build_equal_streams(m,38,39)
        self.build_compressor(m,6)
        self.build_cooler(m,4)
        self.build_compressor(m,7)
        

    
        # ************************************************
        # furnace selection  
        # ************************************************

        self.build_stream_doesnt_exist(m,41)
        self.build_stream_doesnt_exist(m,43)
        self.build_equal_streams(m,40,42)
        self.build_equal_streams(m,44,45)
        self.build_furnace(m,2)

        
        # ************************************************
        # distillation selection  
        # # ************************************************
    
        self.build_distillation(m,3) 
        self.build_distillation(m,5)
        self.build_equal_streams(m,48,50)
        self.build_equal_streams(m,62,63)
        self.build_equal_streams(m,60,69)
        self.build_equal_streams(m,61,70)
        self.build_equal_streams(m,59,66)
        self.build_stream_doesnt_exist(m,49)
        self.build_stream_doesnt_exist(m,51)
        self.build_stream_doesnt_exist(m,52)
        self.build_stream_doesnt_exist(m,53)
        self.build_stream_doesnt_exist(m,55)
        self.build_stream_doesnt_exist(m,56)
        self.build_stream_doesnt_exist(m,54)
    
        # why? this is confusing
        m.stream_split = q = pe.ConstraintList()
        for comp in m.components:
            q.add(m.component_stream_flowrate[57,comp] == m.component_stream_flowrate[59,comp] + m.component_stream_flowrate[60,comp])
        q.add(m.pressure[57] == m.pressure[59])
        q.add(m.temperature[57] == m.temperature[59])
        

        
        # # ************************************************
        # # reactor selection 
        # # ************************************************
       
        m.flash_vapor_constraintlist = f = pe.ConstraintList() 
        for comp in m.components:
            f.add(expr = m.component_stream_flowrate[25,comp] == m.component_stream_flowrate[27,comp])
            

        self.build_stream_doesnt_exist(m,3)
        self.build_equal_streams(m,4,11) 
        
        
        f.add(m.temperature[27] == m.temperature[25])
        f.add(m.pressure[27] == m.pressure[25]) 

        m.reactor_selection_list = g = pe.ConstraintList()
        for comp in m.components:
            g.add(m.component_stream_flowrate[5,comp] + m.component_stream_flowrate[6,comp] == m.component_stream_flowrate[1,comp])
            g.add(m.component_stream_flowrate[5,comp] == m.component_stream_flowrate[1,comp] * m.split_fraction[5])
        g.add(m.temperature[1] == m.temperature[5]) 
        g.add(m.pressure[19] == m.pressure[6]) #?? why P19
        g.add(m.pressure[1] == m.pressure[5])
        
        self.build_heater(m,3)
        self.build_cooler(m,3)
        self.build_flash(m,1)
        self.build_distillation(m,1)
        self.build_stream_doesnt_exist(m,28)
        m.stream_before_oxychlorination_reactor_constranitlist = y = pe.ConstraintList()
        for comp in m.components:
            y.add(m.component_stream_flowrate[21,comp] + m.component_stream_flowrate[19,comp] + m.component_stream_flowrate[68,comp] == m.component_stream_flowrate[22,comp])
        y.add(m.temperature[21] == m.temperature[22])
        y.add(m.temperature[19] == m.temperature[22])
        y.add(m.temperature[68] == m.temperature[22])
        y.add(m.pressure[21] == m.pressure[22])
        y.add(m.pressure[19] == m.pressure[22])
        y.add(m.pressure[68] == m.pressure[22])
        self.build_compressor(m,4)
        for comp in m.components:
            g.add(m.component_stream_flowrate[10,comp] + m.component_stream_flowrate[30,comp] == m.component_stream_flowrate[32,comp])
        g.add(m.temperature[10] == m.temperature[32])
        g.add(m.temperature[30] == m.temperature[32])
        g.add(m.pressure[10] == m.pressure[32])
        g.add(m.pressure[32] >= m.pressure[30])

        self.build_oxychlorination_reactor(m,1) # 1st disjunct ends here
        m.join_inlet_reactor_ = c = pe.ConstraintList()
        for comp in m.components:
            c.add(m.component_stream_flowrate[20,comp] == m.component_stream_flowrate[6,comp] + m.component_stream_flowrate[29,comp]) # Flow match for the reflux stream right before go into reactor
        c.add(m.pressure[20] == m.pressure[6]) # P match for the reflux stream right before go into reactor
        c.add(m.pressure[20] == m.pressure[29])
        c.add(m.temperature[20] == (m.stream_flowrate[6] * m.temperature[6] + m.stream_flowrate[29] * m.temperature[29])/m.stream_flowrate[20]) # T match for the reflux stream right before go into reactor

        self.build_heater(m,1)
        self.build_direct_chlorination_reactor(m,1)
        self.build_cooler(m,1)
        m.join_stream_both_reactor = d = pe.ConstraintList()
        for comp in m.components:
            d.add(m.component_stream_flowrate[8,comp] == m.component_stream_flowrate[2,comp] + m.component_stream_flowrate[7,comp])
        d.add(m.pressure[7] == m.pressure[8])
        d.add(m.temperature[7] == m.temperature[8])

        self.build_stream_doesnt_exist(m,13)
        self.build_stream_doesnt_exist(m,15)
        self.build_stream_doesnt_exist(m,16)
        self.build_stream_doesnt_exist(m,17)
        self.build_equal_streams(m,11,12)
        self.build_equal_streams(m,14,18)
        self.build_compressor(m,1)

        
        
        # ************************************************
        # none selection equipment in flowsheet 
        # ************************************************

        self.build_heater(m,5) 
        self.build_heater(m,2)
        self.build_compressor(m,8)
        self.build_pump(m,2)
        self.build_heater(m,4)
        self.build_cooler(m,5)
        self.build_pump(m,1)
        self.build_valve(m,1)
        
        # ************************************
        # Objective
        # ************************************

        def profit(m):
            return -( 11840 * m.stream_flowrate[70] + 1000 * m.stream_flowrate[69] \
                - 6055 * m.stream_flowrate[1]  - 1000 * m.stream_flowrate[71]) \
                + 1000 * (m.slack_variable_1_P + m.slack_variable_1_N + m.slack_variable_2_P + m.slack_variable_2_N + m.slack_variable_3_P + m.slack_variable_3_N) \
                + sum((m.costelec * m.electricity_requirement[i] + m.compressor_cost[i]) for i in m.compressor)\
                + sum(m.cost_of_flash[i] for i in m.flash)\
                + sum(m.cost_of_distillation[i] for i in m.distillation_column)\
                + sum(m.cost_of_oxychlorination_reactor[i]for i in m.oxychlorination_reactor) \
                + sum(m.cost_of_direct_chlorination_reactor[i] for i in m.direct_chlorination_reactor) \
                + sum(m.furnace_cost[i]for i in m.pyrolysis_furnance) \
                + sum((m.costqc * m.utility_requirement_c[i] + m.cooler_cost[i]) for i in m.cooler)\
                + sum((m.costqh * m.utility_requirement_h[i] + m.heater_cost[i]) for i in m.heater) 
        m.Obj = pe.Objective(rule = profit ,sense = pe.minimize)

    # ************************************************
    # streams define -- functions 
    # ************************************************
    def build_stream_doesnt_exist(self,block,stream): ## this is just a test,change later
        m = self.model
        b = pe.Block()
        setattr(block, 'stream_doesnt_exist_' + str(stream), b)
        b.zero_flow_con = pe.Constraint(expr=m.stream_flowrate[stream] == 0)
        def _zero_component_flows(_b, _c):
            return m.component_stream_flowrate[stream, _c] == 0
        b.zero_component_flows_con = pe.Constraint(m.components, rule=_zero_component_flows)

        def _fixed_temp_con(_b):
            return m.temperature[stream] == 0
        b.fixed_temp_con = pe.Constraint(rule=_fixed_temp_con)

        def _fixed_pressure_con(_b):
            return m.pressure[stream] == 0
        b.fixed_pressure_con = pe.Constraint(rule=_fixed_pressure_con)

    def build_equal_streams(self,block,in_stream,out_stream):
        m = self.model
        t = m.temperature
        p = m.pressure
        b = pe.Block()
        setattr(block, 'equal_streams_' + str(in_stream) + '_' + str(out_stream), b)
        def _component_balances(_b, _c):
            return m.component_stream_flowrate[out_stream, _c] == m.component_stream_flowrate[in_stream, _c]
        b.component_balances = pe.Constraint(m.components, rule=_component_balances)

        def _temp_con(_b):
            return t[in_stream] == t[out_stream]
        b.temp_con = pe.Constraint(rule=_temp_con)

        def _pressure_con(_b):
            return p[in_stream] == p[out_stream]
        b.pressure_con = pe.Constraint(rule=_pressure_con)

    def build_compressor(self,block,unit_number):
        u = unit_number
        m = self.model
        in_stream = self.inlet_streams['compressor'][u]
        out_stream = self.outlet_streams['compressor'][u]
        b = pe.Block()
        setattr(block, 'compressor_'+str(u), b)
        b.pressure_ratio = pe.Var(domain = pe.Reals,doc = ' outlet to inlet',bounds = (0,None))
        ini_pressure_ratio = [1.43927,0,0,1,0,1.59198,1.12511,1]
        b.pressure_ratio = ini_pressure_ratio[u-1]
        
        def mass_balance(b_,_c):
            return m.component_stream_flowrate[in_stream,_c] == m.component_stream_flowrate[out_stream,_c]
        b.Mass_balance =  pe.Constraint(m.components,rule = mass_balance)
        def heat_balance(b_):
            return m.temperature[in_stream] * b.pressure_ratio == m.temperature[out_stream]
        b.Heat_balance = pe.Constraint(rule = heat_balance)
        def energy_balance(b_):
            return m.electricity_requirement[u] == self.alpha * (b.pressure_ratio - 1) * (1/self.compeff) * (self.gamma/(self.gamma - 1)) * 100 * m.temperature[in_stream] * m.stream_flowrate[in_stream] / 60
        b.Energy_balance = pe.Constraint(rule = energy_balance)
        def pressure_ratio_calculation(b_):
            return b.pressure_ratio == (m.pressure[out_stream] / (m.pressure[in_stream])) ** ((self.gamma -1)/self.gamma)
        b.Pressure_ratio_calculation = pe.Constraint(rule = pressure_ratio_calculation) 
        def compressor_cost(b_):
            return m.compressor_cost[u] == m.marshall_swift_index / 280 * ( 50.5 * (m.electricity_requirement[u] + 0.001 ) ** 0.706 + 1)
        b.Compressor_cost = pe.Constraint(rule = compressor_cost)

    def build_cooler(self,block,unit_number):
        u = unit_number
        m = self.model
        in_stream = self.inlet_streams['cooler'][u]
        outlet_stream = self.outlet_streams['cooler'][u]
        b = pe.Block()
        setattr(block, 'cooler_'+str(u), b)
        ini_cooler_area = [0.363711,1,0.244148,0.726812,0]
        b.cooler_area = pe.Var(domain = pe.Reals,bounds = (0,None))
        b.cooler_area = ini_cooler_area[u-1]
        def mass_balance(b_,_c):
            return m.component_stream_flowrate[in_stream,_c] == m.component_stream_flowrate[outlet_stream,_c] 
        b.Mass_balance = pe.Constraint(m.components,rule = mass_balance)
        def heat_balance(b_):
            return (m.temperature[in_stream] * 100 * self.Stream_CP[in_stream -1] * m.stream_flowrate[in_stream] - m.temperature[outlet_stream] * 100 * self.Stream_CP[outlet_stream -1] * m.stream_flowrate[outlet_stream] ) * 3600. * 8500. * 1.0E-12 / 60. == m.utility_requirement_c[u]
        b.Heat_balance = pe.Constraint(rule = heat_balance)
        def pressure_relation(b_):
            return m.pressure[in_stream] == m.pressure[outlet_stream]
        b.Pressure_relation = pe.Constraint(rule = pressure_relation)
        def area_calculation(b_):
            return m.utility_requirement_c[u] == b.cooler_area * (0.001 + (m.temperature[in_stream]-m.temperature[outlet_stream])/2) #?? -3
        b.Area_calculation = pe.Constraint(rule = area_calculation)
        def temp_relation(b_):
            return m.temperature[in_stream] >= m.temperature[outlet_stream]
        b.temperature_relation = pe.Constraint(rule = temp_relation)
        def cost_of_cooler(b_):
            return m.cooler_cost[u] == m.marshall_swift_index / 280 * (3.17 * ( b.cooler_area + 0.001) ** 0.641 + 0.75) #?? +0.75
        b.Cost_of_cooler = pe.Constraint(rule = cost_of_cooler)

    def build_furnace(self,block,unit_number):
        u = unit_number 
        m = self.model
        in_stream = self.inlet_streams['furnace'][u]
        out_stream = self.outlet_streams['furnace'][u]
        b = pe.Block()
        setattr(block, 'furnace_'+str(u), b)
        b.conversion = pe.Var(within = pe.Reals,bounds = (0.5,0.6),initialize = 0.5616)
        b.consumption = pe.Var(within = pe.Reals,bounds = (0,20),initialize = 17.8341)
        b.furnace_pressure = pe.Var(within = pe.PositiveReals,bounds = (25,30),initialize = 25)
        b.furnace_temperature = pe.Var(within = pe.PositiveReals,bounds =(7.730,8.23),initialize = 7.8)
        b.furnace_volume = pe.Var(domain = pe.PositiveReals,bounds = (0,100),initialize = 36.49)
        def conversion_rate_calculation(b_):
            return b.conversion == m.conversion_corelation[in_stream] * m.temperature[in_stream] * 100
        b.Conversion_rate = pe.Constraint(rule = conversion_rate_calculation)
        def consumption_calculation(b_):
            return b.consumption == b.conversion * m.component_stream_flowrate[in_stream,'ethylene di-chloride']
        b.Consumption_calculation = pe.Constraint(rule = consumption_calculation)
        def pressure_relation(b_):
            return m.pressure[in_stream] == m.pressure[out_stream]
        b.Pressure_relation = pe.Constraint(rule = pressure_relation)
        def temperature_relation(b_):
            return m.temperature[in_stream]  == m.temperature[out_stream]
        b.Temperature_relation = pe.Constraint(rule = temperature_relation)
        def furnace_P_set_up(b_):
            if in_stream == 41:
                return m.pressure[in_stream] == b.furnace_pressure
            if in_stream == 42:
                return m.pressure[in_stream] <= b.furnace_pressure
            else:
                return pe.Constraint.Skip 
        b.Furnace_P_set_up = pe.Constraint(rule = furnace_P_set_up )
        def furnace_T_set_up(b_):
            if in_stream == 41:
                return m.temperature[in_stream] == b.furnace_temperature
            if in_stream == 42:
                return m.temperature[in_stream] >= b.furnace_temperature
            else:
                return pe.Constraint.Skip 
        b.Furnace_T_set_up = pe.Constraint(rule =furnace_T_set_up)
        def vinyl_calculation(b_):
            return m.component_stream_flowrate[out_stream,'vinyl chloride'] == b.consumption
        b.Vinyl_calculation = pe.Constraint(rule = vinyl_calculation)
        def hydrogen_chloride(b_):
            return m.component_stream_flowrate[out_stream,'hydrogen chloride'] == b.consumption
        b.Hydrogen_chloride = pe.Constraint(rule = hydrogen_chloride)
        def ethylene_dichloride(b_):
            return m.component_stream_flowrate[out_stream,'ethylene di-chloride'] == m.component_stream_flowrate[in_stream,'ethylene di-chloride'] - b.consumption
        b.Ethylene_dichloride = pe.Constraint(rule = ethylene_dichloride)
        def pyrolysis_volume(b_):
            return b.furnace_volume == m.upper_bound_consumption[u] * 0.082 * 7.5 * 100 / m.upper_bound_pressure[u]
        b.Pyrolysis_volume = pe.Constraint(rule = pyrolysis_volume)
        def cost_furnace(b_):
            return m.furnace_cost[u] == m.marshall_swift_index / 280 * m.material_of_construction_factor_f[u] * m.material_of_construction_factor * (5.2 * (b.furnace_volume + 0.001) ** 0.6 + 0.75)  
        b.Cost_furnace = pe.Constraint(rule = cost_furnace)
    def build_distillation(self,block,unit_number):
        u = unit_number
        m = self.model
        in_stream = self.inlet_streams['distillation'][u]
        vapor_outlet = self.vapor_outlets['distillation'][u]
        liquid_outlet = self.liquid_outlets['distillation'][u]
        heavy_component = self.heavy_components['distillation'][u]
        heavy_key_component = self.heavy_key_components['distillation'][u]
        light_component = self.light_components['distillation'][u]
        light_key_component = self.light_key_components['distillation'][u]
        key_component_list = self.key_components['distillation'][u]
        b = pe.Block()
        setattr(block, 'distillation_'+str(u), b)
        b.min_number_of_tray = pe.Var(domain = pe.PositiveReals,bounds = (2,199))
        b.number_of_trays = pe.Var(domain = pe.Reals,bounds = (0,30))
        b.min_reflux_ratio = pe.Var(domain = pe.PositiveReals,bounds = (0.001,30))
        b.reflux_ratio = pe.Var(domain = pe.Reals,bounds = (0,30),initialize = 0)
        b.average_volatility = pe.Var(domain = pe.PositiveReals,bounds = (1.1,100))
        b.column_pressure = pe.Var(domain = pe.PositiveReals,bounds = (1,30))
        ini_average_volatility = [8.0931,0,50.2248,0,12.2081]
        ini_column_pressure = [1,1,4.84493,1,1]
        ini_number_of_trays = [39.4235,0,20.3567,0,38.2632]
        ini_min_number_of_tray = [9.85588,2,5.08917,2,9.5658]
        ini_reflux_ratio = [0.188151,0,0.067786,0,0.190702]
        ini_min_reflux_ratio = [0.156792,0,0.0564883,0,0.158919]
        b.column_pressure = ini_column_pressure[u-1]
        b.average_volatility = ini_average_volatility[u-1]
        b.number_of_trays = ini_number_of_trays[u-1] 

        b.min_number_of_tray = ini_min_number_of_tray[u-1]
        b.min_reflux_ratio = ini_min_reflux_ratio[u-1]
        b.reflux_ratio = ini_reflux_ratio[u-1]
        def mass_balance(b_,_c):
            return m.component_stream_flowrate[in_stream,_c] ==  m.component_stream_flowrate[vapor_outlet,_c] + m.component_stream_flowrate[liquid_outlet,_c]
        b.Mass_balance = pe.Constraint(m.components,rule = mass_balance)
        def bottom_vapor_pressure_corraltion(b_,_c):
            if _c in key_component_list:
                antonie_A = self.antoine[_c][0]
                antonie_B = self.antoine[_c][1]
                antonie_C = self.antoine[_c][2]
                return m.vapor_pressure[liquid_outlet,_c] * 7500.6168 == pe.exp(antonie_A - antonie_B  /(m.temperature[liquid_outlet] * 100 + antonie_C)) + m.slack_variable_1_P - m.slack_variable_1_N
            else:
                return pe.Constraint.Skip
        b.Bottom_vapor_pressure_corraltion = pe.Constraint(m.components, rule = bottom_vapor_pressure_corraltion)
        def top_vapor_pressure_corraltion(b_,_c):
            if _c in key_component_list:
                antonie_A = self.antoine[_c][0]
                antonie_B = self.antoine[_c][1]
                antonie_C = self.antoine[_c][2]
                return m.vapor_pressure[vapor_outlet,_c] * 7500.6168  == pe.exp(antonie_A - antonie_B /(m.temperature[vapor_outlet] * 100 + antonie_C)) + m.slack_variable_2_P - m.slack_variable_2_N
            else:
                return pe.Constraint.Skip
        b.Top_vapor_pressure_corraltion = pe.Constraint(m.components, rule = top_vapor_pressure_corraltion)
        def average_relative_volatility(b_):
            return 0.5 * (m.vapor_pressure[liquid_outlet,light_key_component] / m.vapor_pressure[liquid_outlet,heavy_key_component]) * (m.vapor_pressure[vapor_outlet,light_key_component] / m.vapor_pressure[vapor_outlet,heavy_key_component]) + m.slack_variable_3_P - m.slack_variable_3_N == b.average_volatility
        b.Average_relative_volatility = pe.Constraint(rule = average_relative_volatility)
        def underwood_equation(b_): 
            return m.component_stream_flowrate[in_stream,light_key_component]  * b.min_reflux_ratio * (b.average_volatility -1) == m.stream_flowrate[in_stream]
        b.Underwood_equation = pe.Constraint(rule = underwood_equation)
        def actual_reflux_calculation(b_):
            return b.reflux_ratio == 1.2 * b.min_reflux_ratio 
        b.Actual_reflux_calculation = pe.Constraint(rule = actual_reflux_calculation)
        def fenske_equation(b_):
            return b.min_number_of_tray * pe.log(b.average_volatility) == pe.log((m.stream_flowrate[vapor_outlet] + self.eps1) /(m.component_stream_flowrate[vapor_outlet,heavy_key_component] + self.eps1) * (m.stream_flowrate[liquid_outlet] + self.eps1) /(m.component_stream_flowrate[liquid_outlet,light_key_component] + self.eps1))
        b.Fenske_equation = pe.Constraint(rule = fenske_equation)
        def actual_tray_number(b_):
            return b.number_of_trays == b.min_number_of_tray * 2 / self.disteff
        b.Actual_tray_number = pe.Constraint(rule = actual_tray_number)
        def recovery_specification(b_):
            return m.component_stream_flowrate[vapor_outlet,heavy_key_component] <= 0.05 * m.component_stream_flowrate[in_stream,heavy_key_component]
        b.Recovery_specification = pe.Constraint(rule = recovery_specification)
        def heavy_component_calculation(b_):
            return m.component_stream_flowrate[liquid_outlet,heavy_component]  == m.component_stream_flowrate[in_stream,heavy_component]
        b.Heavy_component_calculation = pe.Constraint(rule = heavy_component_calculation)
        def light_component_calculation(b_):
            return  m.component_stream_flowrate[vapor_outlet,light_component]  == m.component_stream_flowrate[in_stream,light_component]
        b.Light_component_calculation = pe.Constraint(rule = light_component_calculation)
        def inlet_pressure_relation(b_):
            return b.column_pressure <= m.pressure[in_stream]
        b.Inlet_pressure_relation = pe.Constraint(rule = inlet_pressure_relation)
        def bottom_vapor_pressure_relation(b_):
            return b.column_pressure ==  m.vapor_pressure[liquid_outlet,heavy_key_component]
        b.Bottom_vapor_pressure_relation = pe.Constraint(rule = bottom_vapor_pressure_relation)
        def top_vapor_pressure_relation(b_): # changed 12/15
            if u != 1:
                return b.column_pressure == m.vapor_pressure[vapor_outlet,light_key_component]
            else:
                return pe.Constraint.Skip
        b.Top_vapor_pressure_relation = pe.Constraint(rule = top_vapor_pressure_relation)
        def liquid_outlet_pressure_relation(b_):
            return b.column_pressure == m.pressure[liquid_outlet]
        b.Liquid_outlet_pressure_relation = pe.Constraint(rule = liquid_outlet_pressure_relation)
        def vapor_outlet_pressure_relation(b_):
            return b.column_pressure == m.pressure[vapor_outlet]
        b.Vapor_outlet_pressure_relation = pe.Constraint(rule = vapor_outlet_pressure_relation)
        def distillation_cost_calc(b_):
            return m.cost_of_distillation[u] == b.number_of_trays  * m.marshall_swift_index / 280 * m.pressure_factor * 1.7 * (0.189 * m.average_flow[u] + 3)
        b.Distillation_cost_calc = pe.Constraint(rule = distillation_cost_calc)
    def build_pump(self,block,unit_number):
        u = unit_number
        m = self.model
        in_stream = self.inlet_streams['pump'][u]
        outlet_stream = self.outlet_streams['pump'][u]
        b = pe.Block()
        setattr(block, 'pump_'+str(u), b)
        def mass_balance(b_,_c):
            return m.component_stream_flowrate[in_stream,_c] == m.component_stream_flowrate[outlet_stream,_c] 
        b.Mass_balance = pe.Constraint(m.components,rule = mass_balance)
        def pressure_relation(b_):
            return m.pressure[in_stream] <= m.pressure[outlet_stream]
        b.Pressure_relation = pe.Constraint(rule = pressure_relation)
        def temp_relation(b_):
            return m.temperature[in_stream] == m.temperature[outlet_stream]
        b.temperature_relation = pe.Constraint(rule = temp_relation)
    def build_valve(self,block,unit_number):
        u = unit_number
        m = self.model
        in_stream = self.inlet_streams['valve'][u]
        outlet_stream = self.outlet_streams['valve'][u]
        b = pe.Block()
        setattr(block, 'valve_'+str(u), b)

        def mass_balance(b_,_c):
            return m.component_stream_flowrate[in_stream,_c] == m.component_stream_flowrate[outlet_stream,_c] 
        b.Mass_balance = pe.Constraint(m.components,rule = mass_balance)
        def pressure_relation(b_):
            return m.pressure[in_stream] >= m.pressure[outlet_stream]
        b.Pressure_relation = pe.Constraint(rule = pressure_relation)
        def temp_relation(b_):
            return m.temperature[in_stream] / (m.pressure[in_stream] ** ((self.gamma-1)/self.gamma)) == m.temperature[outlet_stream] / (m.pressure[outlet_stream] ** ((self.gamma-1)/self.gamma))
        b.temperature_relation = pe.Constraint(rule = temp_relation)
        
    def build_heater(self,block,unit_number):
        u = unit_number
        m = self.model
        in_stream = self.inlet_streams['heater'][u]
        outlet_stream = self.outlet_streams['heater'][u]
        b = pe.Block()
        setattr(block, 'heater_'+str(u), b)
        ini_heater_area = [0.0450848,0.0113322,0,0.567691,0.0726877]
        b.heater_area = pe.Var(domain = pe.Reals,bounds = (0,None))
        b.heater_area = ini_heater_area[u-1]
        def mass_balance(b_,_c):
            return m.component_stream_flowrate[in_stream,_c] == m.component_stream_flowrate[outlet_stream,_c]
        b.Mass_balance = pe.Constraint(m.components,rule = mass_balance)
        def heat_balance(b_):
            return ( m.temperature[outlet_stream] * 100 * self.Stream_CP[outlet_stream - 1] * m.stream_flowrate[outlet_stream] - m.temperature[in_stream] * 100 * self.Stream_CP[in_stream - 1] * m.stream_flowrate[in_stream]) * 3600. * 8500. * 1.0E-12 / 60. == m.utility_requirement_h[u]
        b.Heat_balance = pe.Constraint(rule = heat_balance)
        def pressure_relation(b_):
            return m.pressure[in_stream] == m.pressure[outlet_stream]
        b.Pressure_relation = pe.Constraint(rule = pressure_relation)
        def area_calculation(b_):
            return m.utility_requirement_h[u] == b.heater_area * (0.001 + ( - m.temperature[in_stream] + m.temperature[outlet_stream])/2)
        b.Area_calculation = pe.Constraint(rule = area_calculation)
        # def temp_relation(b_):
        #     return m.temperature[in_stream] <= m.temperature[outlet_stream]
        # b.temperature_relation = pe.Constraint(rule = temp_relation)
        def cost_of_heater(b_):
            return m.heater_cost[u] == m.marshall_swift_index / 280 * (3.17 * ( b.heater_area + 0.001) ** 0.641 + 0.75)
        b.Cost_of_heater = pe.Constraint(rule = cost_of_heater)

    def build_flash(self,block,unit_number): # ?? careful about binary == 0 
        u = unit_number
        m = self.model
        in_stream = self.inlet_streams['flash'][u]
        vapor_outlet = self.vapor_outlets['flash'][u]
        liquid_outlet = self.liquid_outlets['flash'][u]
        flash_key_component = self.flash_key_components['flash'][u]
        b = pe.Block()
        setattr(block, 'flash_'+str(u), b)
        b.flash_vapor_phase_recovery = pe.Var(m.components,domain = pe.NonNegativeReals,initialize = 0)
        b.flash_temp = pe.Var(domain = pe.PositiveReals,doc = 'unit: 100k',bounds = (2.5,5.5),initialize = 5.29032)
        b.flash_pressure = pe.Var(domain = pe.PositiveReals,doc = 'unit: maga-pascal',bounds = (2.5,6.5),initialize = 4.84493)
        def mass_balance(b_,_c):
            return m.component_stream_flowrate[in_stream,_c] == m.component_stream_flowrate[vapor_outlet,_c] + m.component_stream_flowrate[liquid_outlet,_c]
        b.Mass_balance = pe.Constraint(m.components,rule = mass_balance)
        def vapor_pressure_corraltion(b_,_c):
            antonie_A = self.antoine[_c][0]
            antonie_B = self.antoine[_c][1]
            antonie_C = self.antoine[_c][2]
            return pe.log(m.vapor_pressure[liquid_outlet,_c] * 7500.6168) == antonie_A - antonie_B /(m.temperature[liquid_outlet] * 100 + antonie_C)
        b.Vapor_pressure_corraltion = pe.Constraint(m.components,rule = vapor_pressure_corraltion)
        def flash_vapor_recovery_relation(b_,_c): 
            return b.flash_vapor_phase_recovery[flash_key_component]  * (b.flash_vapor_phase_recovery[_c] * m.vapor_pressure[liquid_outlet,flash_key_component] + (1 - b.flash_vapor_phase_recovery[_c]) * m.vapor_pressure[liquid_outlet,_c]) == m.vapor_pressure[liquid_outlet,flash_key_component] * b.flash_vapor_phase_recovery[_c]
        b.Flash_vapor_recovery_relation = pe.Constraint(m.components,rule = flash_vapor_recovery_relation)
        def equilibrium_relation(b_,_c):
            return m.component_stream_flowrate[vapor_outlet,_c] == m.component_stream_flowrate[in_stream,_c] * b.flash_vapor_phase_recovery[_c]
        b.Equilibrium_relation = pe.Constraint(m.components,rule = equilibrium_relation)
        def pressure_relation(b_):
            return b.flash_pressure * m.stream_flowrate[liquid_outlet] == sum(m.vapor_pressure[liquid_outlet,i] * m.component_stream_flowrate[liquid_outlet,i] for i in m.components)
        b.Pressure_relation = pe.Constraint(rule = pressure_relation)
        def inlet_pressure_relation(b_):
            return b.flash_pressure == m.pressure[in_stream]
        b.Inlet_pressure_relation = pe.Constraint(rule = inlet_pressure_relation)
        def liquid_outlet_pressure_relation(b_):
            return b.flash_pressure == m.pressure[liquid_outlet]
        b.Liquid_outlet_pressure_relation = pe.Constraint(rule = liquid_outlet_pressure_relation)
        def vapor_outlet_pressure_relation(b_):
            return b.flash_pressure == m.pressure[vapor_outlet]
        b.Vapor_outlet_pressure_relation = pe.Constraint(rule = vapor_outlet_pressure_relation)
        def inlet_temp_relation(b_):
            return b.flash_temp == m.temperature[in_stream]
        b.Inlet_temp_relation = pe.Constraint(rule = inlet_temp_relation)
        def liquid_outlet_temp_relation(b_):
            return b.flash_temp == m.temperature[liquid_outlet]
        b.Liquid_outlet_temp_relation = pe.Constraint(rule = liquid_outlet_temp_relation)
        def vapor_outlet_temp_relation(b_):
            return b.flash_temp == m.temperature[vapor_outlet]
        b.Vapor_outlet_temp_relation = pe.Constraint(rule = vapor_outlet_temp_relation)
        def cost_flash_calc(b_):
            return m.cost_of_flash[u] == m.marshall_swift_index / 280 * (0.240 * m.residence_time ** 0.738 * ( m.stream_flowrate[in_stream] + 0.01) ** 0.738 + 1.0)
        b.Cost_flash_calc = pe.Constraint(rule = cost_flash_calc)
    def build_direct_chlorination_reactor(self,block,unit_number):
        u = unit_number
        m = self.model
        in_stream = self.inlet_streams['direct'][u]
        outlet_stream = self.outlet_streams['direct'][u]
        b = pe.Block()
        setattr(block, 'direct_chlorination_reactor_'+str(u), b)
        b.react_constant = pe.Var(domain = pe.PositiveReals,initialize = 0.000000007246069)
        b.conversion_of_key_component = pe.Var(domain = pe.PositiveReals,initialize =1)
        b.key_consumption_rate = pe.Var(domain = pe.PositiveReals,bounds = (0,50),initialize = 8.91704)
        b.reactor_density = pe.Var(domain = pe.PositiveReals, doc = 'unit:Kmol/m^3',initialize = 0.236585)
        b.residence_time = pe.Var(domain = pe.PositiveReals, doc = 'unit: second',initialize = 2,bounds = (2,20))
        b.reactor_volume = pe.Var(domain = pe.PositiveReals, doc = 'unit: cubic meter',initialize = 150.762,bounds = (0,200))
        b.heat_removed = pe.Var(domain = pe.PositiveReals,initialize = 828.648,bounds = (0,30000))
        def inlet_stream_match(b_):
            return m.component_stream_flowrate[in_stream,'ethylene'] == m.component_stream_flowrate[in_stream,'chlorine']
        b.Inlet_stream_match = pe.Constraint(rule = inlet_stream_match)
        def react_constant_calculation(b_):
            return b.react_constant == 4.3e10 * pe.exp(-14265 / (m.temperature[in_stream] * 100))
        b.React_constant_calculation = pe.Constraint(rule = react_constant_calculation)
        def key_conversion_calculation(b_):
            return b.conversion_of_key_component == 1 - 0.45 * b.react_constant * m.pressure[in_stream] * m.stream_flowrate[in_stream] / 60
        b.Key_conversion_calculation = pe.Constraint(rule = key_conversion_calculation)
        def consumption_rate_calculation(b_):
            return b.key_consumption_rate == b.conversion_of_key_component * m.component_stream_flowrate[in_stream,'ethylene']
        b.Consumption_rate_calculation = pe.Constraint(rule = consumption_rate_calculation)
        def outlet_match_1(b_):
            return b.key_consumption_rate == m.component_stream_flowrate[outlet_stream,'ethylene di-chloride']
        b.Outlet_match_1 = pe.Constraint(rule = outlet_match_1)
        def outlet_match_2(b_,_c):
            if _c != 'ethylene di-chloride':
                return 0 == m.component_stream_flowrate[outlet_stream,_c]
            else:
                return pe.Constraint.Skip
        b.Outlet_match_2 = pe.Constraint(m.components, rule = outlet_match_2)
        def pressure_relation(b_):
            return m.pressure[in_stream] == m.pressure[outlet_stream]
        b.Pressure_relation = pe.Constraint(rule = pressure_relation)
        def temp_relation(b_):
            return m.temperature[in_stream] <= m.temperature[outlet_stream]
        b.Temp_relation = pe.Constraint(rule = temp_relation) 
        def reactor_density_calculation(b_):
            return b.reactor_density * 0.082 * 100 * m.temperature[outlet_stream] == 10 * 0.97 * m.pressure[outlet_stream]
        b.Reactor_density_calculation = pe.Constraint(rule = reactor_density_calculation)
        def residence_time_calculation(b_):
            return 1 - b.conversion_of_key_component == m.component_stream_flowrate[in_stream,'ethylene'] * pe.exp(- b.conversion_of_key_component * b.residence_time * 3 / b.reactor_density ) 
        b.Residence_time_calculation = pe.Constraint(rule = residence_time_calculation)
        def reactor_volume_calculation(b_):
            return b.reactor_volume * b.reactor_density == b.residence_time * m.stream_flowrate[in_stream]
        b.Reactor_volume_calculation = pe.Constraint(rule = reactor_volume_calculation)
        def energy_balance(b_):
            return b.heat_removed == abs(m.heat_of_reaction_d[u]) * b.key_consumption_rate * 8500 * 60 * 1e-9
        b.Energy_balance = pe.Constraint(rule = energy_balance)
        def temp_calculation(b_):
            return m.temperature[in_stream] + b.heat_removed * 1e+9 / (3 * self.Stream_CP[8] * 2 * 60 * 100 * 8500) == m.temperature[outlet_stream] 
        b.Temp_calculation = pe.Constraint(rule = temp_calculation)
        def cost_of_direct_calc(b_):
            return m.cost_of_direct_chlorination_reactor[u] == m.marshall_swift_index / 280 * m.material_of_construction_factor * m.pressure_factor * (0.225 * (b.reactor_volume + 0.001) +1) + m.catalyst_replacement * 4 * b.reactor_volume
        b.Cost_of_direct_calc = pe.Constraint(rule = cost_of_direct_calc)
    def build_oxychlorination_reactor(self,block,unit_number):
        u = unit_number
        m = self.model
        in_stream = self.inlet_streams['oxychlorination'][u]
        outlet_stream = self.outlet_streams['oxychlorination'][u]
        b = pe.Block()
        setattr(block, 'oxychlorination_reactor_'+str(u), b)
        b.rate_constant = pe.Var(domain = pe.PositiveReals,bounds = (0.261494,0.647941),initialize =0.261494)
        b.conversion_of_key_component = pe.Var(domain = pe.PositiveReals,initialize = 1)
        b.key_consumption_rate = pe.Var(domain = pe.PositiveReals,bounds = (0.950,100),initialize = 8.91704)
        b.reactor_volume = pe.Var(domain = pe.PositiveReals, doc = 'unit: cubic meter',initialize = 146.239,bounds = (0,200))
        b.heat_removed = pe.Var(domain = pe.PositiveReals,initialize = 1088.94,bounds = (0,200000))
        def inlet_stream_match_1(b_):
            return m.component_stream_flowrate[in_stream,'ethylene'] == 0.5 * m.component_stream_flowrate[in_stream,'hydrogen chloride']
        b.Inlet_stream_match_1 = pe.Constraint(rule = inlet_stream_match_1)
        def inlet_stream_match_2(b_):
            return m.component_stream_flowrate[in_stream,'ethylene'] == 2.0 * m.component_stream_flowrate[in_stream,'oxygen']
        b.Inlet_stream_match_2 = pe.Constraint(rule = inlet_stream_match_2)
        def rate_constant_calculation(b_):
            return b.rate_constant == 5.8e+12 * pe.exp( -15150 / (m.temperature[in_stream] * 100))
        b.Rate_constant_calculation = pe.Constraint(rule = rate_constant_calculation)   
        def key_conversion_calculation(b_):
            return b.conversion_of_key_component >= 1 - 0.02 / pe.exp(b.rate_constant) 
        b.Key_conversion_calculation = pe.Constraint(rule = key_conversion_calculation)
        def consumption_rate_calculation(b_):
            return b.key_consumption_rate == b.conversion_of_key_component * m.component_stream_flowrate[in_stream,'ethylene']
        b.Consumption_rate_calculation = pe.Constraint(rule = consumption_rate_calculation)
        def outlet_match_1(b_):
            return b.key_consumption_rate == m.component_stream_flowrate[outlet_stream,'ethylene di-chloride']
        b.Outlet_match_1 = pe.Constraint(rule = outlet_match_1)
        def outlet_match_2(b_):
            return m.component_stream_flowrate[outlet_stream,'ethylene'] == m.component_stream_flowrate[in_stream,'ethylene'] - b.key_consumption_rate
        b.Outlet_match_2 = pe.Constraint(rule = outlet_match_2)
        def outlet_match_3(b_):
            return m.component_stream_flowrate[outlet_stream,'hydrogen chloride'] == m.component_stream_flowrate[in_stream,'hydrogen chloride'] - 2.0 * b.key_consumption_rate
        b.Outlet_match_3 = pe.Constraint(rule = outlet_match_3)
        def outlet_match_4(b_):
            return m.component_stream_flowrate[outlet_stream,'oxygen'] == m.component_stream_flowrate[in_stream,'oxygen'] - 0.5 * b.key_consumption_rate
        b.Outlet_match_4 = pe.Constraint(rule = outlet_match_4)
        def pressure_relation(b_):
            return m.pressure[in_stream] == m.pressure[outlet_stream]
        b.Pressure_relation = pe.Constraint(rule = pressure_relation)
        def reactor_volume_calculation(b_):
            return b.reactor_volume == b.key_consumption_rate * 0.082 * 5.0 * 100 / 2.5
        b.Reactor_volume_calculation = pe.Constraint(rule = reactor_volume_calculation)
        def energy_balance(b_):
            return b.heat_removed == abs(m.heat_of_reaction_o[u]) * b.key_consumption_rate * 8500 * 60 * 1e-9
        b.Energy_balance = pe.Constraint(rule = energy_balance)
        def temp_relation(b_):
            return m.temperature[in_stream] + b.heat_removed * 1e+9 / (2 * self.Stream_CP[outlet_stream -1 ]  * 100 * 15 * 60 * 8500) == m.temperature[outlet_stream]
        b.Temp_relation = pe.Constraint(rule = temp_relation)
        def cost_oxy_calc(b_):
            return m.cost_of_oxychlorination_reactor[u] == m.marshall_swift_index / 280 * m.material_of_construction_factor * m.pressure_factor * (0.125 * (b.reactor_volume + 0.001) ** 0.738 +1) + m.catalyst_replacement * 4 * b.reactor_volume
        b.Cost_oxy_calc = pe.Constraint(rule = cost_oxy_calc)  


                            



def enumerate_solutions(m):
    # TODO: MODIFY THIS TO REPRESENT THE VYNIL PROBLEM (NOT HDA)
    import time
    
    O2_treatments = ['Air','O2']
    Compressor_selections_in = ['one_stage_in','second_stage_in']
    Reactor_selections = ['direct_reactor','OXY_reactor','both']
    Furnace_selections = ['recycle_membrane','recycle_purge']
    Compressor_selections_out = ['one_stage_out','second_stage_out']
    distillation_selections = ['up','down']
    

    since = time.time()
    for O2_treatment in O2_treatments:
        for Compressor_selection_in in Compressor_selections_in:
            for Reactor_selection in Reactor_selections:
                for Furnace_selection in Furnace_selections:
                    for Compressor_selection_out in Compressor_selections_out:
                        for distillation_selection in distillation_selections:
                            if H2_treatment == 'purify':
                                m.purify_H2.indicator_pe.Var.fix(1)
                                m.no_purify_H2.indicator_pe.Var.fix(0)
                            else:
                                m.purify_H2.indicator_pe.Var.fix(0)
                                m.no_purify_H2.indicator_pe.Var.fix(1)
                            if Reactor_selection == 'adiabatic_reactor':
                                m.adiabatic_reactor.indicator_pe.Var.fix(1)
                                m.isothermal_reactor.indicator_pe.Var.fix(0)
                            else:
                                m.adiabatic_reactor.indicator_pe.Var.fix(0)
                                m.isothermal_reactor.indicator_pe.Var.fix(1)
                            if Methane_recycle_selection == 'recycle_membrane':
                                m.recycle_methane_purge.indicator_pe.Var.fix(0)
                                m.recycle_methane_membrane.indicator_pe.Var.fix(1)
                            else:
                                m.recycle_methane_purge.indicator_pe.Var.fix(1)
                                m.recycle_methane_membrane.indicator_pe.Var.fix(0)
                            if Absorber_recycle_selection == 'yes_absorber':
                                m.absorber_hydrogen.indicator_pe.Var.fix(1)
                                m.recycle_hydrogen.indicator_pe.Var.fix(0)
                            else:
                                m.absorber_hydrogen.indicator_pe.Var.fix(0)
                                m.recycle_hydrogen.indicator_pe.Var.fix(1)
                            if Methane_product_selection == 'methane_column':
                                m.methane_flash_separation.indicator_pe.Var.fix(0)
                                m.methane_distillation_column.indicator_pe.Var.fix(1)
                            else:
                                m.methane_flash_separation.indicator_pe.Var.fix(1)
                                m.methane_distillation_column.indicator_pe.Var.fix(0)
                            if Toluene_product_selection == 'toluene_column':
                                m.toluene_flash_separation.indicator_pe.Var.fix(0)
                                m.toluene_distillation_column.indicator_pe.Var.fix(1)
                            else:
                                m.toluene_flash_separation.indicator_pe.Var.fix(1)
                                m.toluene_distillation_column.indicator_pe.Var.fix(0)
                            opt = SolverFactory('gdpopt')
                            res = opt.solve(m,tee = False,
                                            strategy = 'LOA',
                                            time_limit = 3600,
                                            mip_solver = 'gams',
                                            mip_solver_args= dict(solver = 'gurobi', warmstart=True),
                                            nlp_solver = 'gams',
                                            nlp_solver_args= dict(solver = 'ipopt', add_options = ['option optcr = 0'],warmstart=True),
                                            minlp_solver = 'gams',
                                            minlp_solver_args= dict(solver = 'dicopt', warmstart=True),
                                            subproblem_presolve=False,
                                            init_strategy = 'no_init',
                                            set_cover_iterlim = 20 
                                            )
    #                         print('{0:<30}{1:<30}{2:<30}{3:<30}{4:<30}{5:<30}{6:<30}{7:<30}'.format(H2_treatment, Reactor_selection, Methane_recycle_selection,Absorber_recycle_selection,Methane_product_selection,Toluene_product_selection,str(res.solver.termination_condition),value(m.obj)))
    # time_elapsed = time.time() - since 
    # print('The code run {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

def solve_with_gdpopt(m):
    '''
    This function solves model m using GDPOpt
    '''

    opt = pe.SolverFactory('gdpopt')
    res = opt.solve(m, tee=True,
                    strategy='LOA',
                    # strategy='GLOA',
                    time_limit=3600,
                    mip_solver='gams',
                    mip_solver_args=dict(solver='cplex', warmstart=True),
                    nlp_solver='gams',
                    nlp_solver_args=dict(solver='baron', tee=True, keepfiles = True,tmpdir = 'C:/Users/hithi/Desktop/vinyl/',symbolic_solver_labels=True,
        add_options=[
            'GAMS_MODEL.optfile = 1;' 
        ]),
                    minlp_solver='gams',
                    minlp_solver_args=dict(solver='dicopt', warmstart=True),
                    subproblem_presolve=False,
                    init_strategy='no_init',
                    set_cover_iterlim=20,
                    # calc_disjunctive_bounds=True
                    )
    return res


def infeasible_constraints(m):
    '''
    This function checks infeasible constraint in the model 
    '''
    log_infeasible_constraints(m) 
    

def solve_with_minlp(m):
    # pe.TransformationFactory('gdp.bigm').apply_to(m, bigM=100)
    # TransformationFactory('gdp.chull').apply_to(m)
    result = pe.SolverFactory('gams').solve(
        m, solver='baron', tee=True, keepfiles = True,tmpdir = 'C:/Users/hithi/Desktop/vinyl/',symbolic_solver_labels=True,
        add_options=[
            'GAMS_MODEL.optfile = 1;',
            'option optcr = 0'
        ]
    );
    
    return m
    
if __name__ == "__main__":
    m = VinylChlorideModel().model
    m.stream_flowrate[23].fix(9.91704)
    m.stream_flowrate[24].fix(9.91704)
    m.stream_flowrate[26].fix(9.91704)
    m.stream_flowrate[31].fix(1)
    m.stream_flowrate[50].fix(49.5899)
    m.stream_flowrate[58].fix(31.7434)
    m.stream_flowrate[61].fix(17.8217)
    m.stream_flowrate[60].fix(0.012401)
    m.stream_flowrate[70].fix(17.8217)
    m.component_stream_flowrate[23,'water'].fix(1)
    m.component_stream_flowrate[24,'water'].fix(1)
    m.temperature[11].fix(3)
    m.temperature[12].fix(3)
    m.temperature[14].fix(4.31782)
    m.temperature[48].fix(7.8)
    m.temperature[60].fix(4.05774)
    m.pressure[60].fix(19.1292)
    m.pressure[31].fix(1)
    m.slack_variable_3_P.fix(0)
    m.slack_variable_3_N.fix(0)

    res = solve_with_minlp(m)
    infeasible_constraints(res)

## GDP model 
import pyomo.environ as pe
import pyomo.gdp as gdp
import pandas as pd 
from pyomo.util.infeasible import log_infeasible_constraints
from pyomo.contrib.fbbt.fbbt import fbbt, compute_bounds_on_expr

class VinylChlorideModel(object):
    def __init__(self):
        self.model = m =  pe.ConcreteModel() # main model
        ## parameters
        self.alpha = 0.3665 #compressor coefficient
        self.compeff = 0.750 #compressor efficiency
        self.gamma = 1.3 #ratio of Cp to Cv
        self.disteff = 0.5 #distillation column tray efficiency
        self.eps1 = 1e-4
        ## Sets define 
        m.stream = pe.RangeSet(71)
        m.compressor = pe.RangeSet(8)
        m.distillation_column = pe.RangeSet(5)
        m.flash = pe.RangeSet(1)
        m.pyrolysis_furnance = pe.RangeSet(2)
        m.cooler = pe.RangeSet(5)
        m.heater = pe.RangeSet(5)
        m.pump = pe.RangeSet(2)
        m.valve = pe.RangeSet(1)
        m.feed = pe.RangeSet(2)
        m.direct_chlorination_reactor = pe.RangeSet(1)
        m.oxychlorination_reactor = pe.RangeSet(1)
        m.components = pe.Set(initialize = ['nitrogen','oxygen','ethylene','hydrogen chloride','vinyl chloride','chlorine','ethylene di-chloride','water'])

        # parameter related to costs (in unites of $1000)
        m.costelec = pe.Param(initialize = 0.340,doc ="electricity cost,unit kw-year")
        m.costqc = pe.Param(initialize = 0.70,doc ="cooling cost,unit: 1e9KJ")
        m.costqh = pe.Param(initialize = 8.0,doc = "heating cost,unit: 1e9KJ")
        m.costfuel = pe.Param(initialize = 4.0,doc = "fuel cost furnace,unit: 1e9KJ")
        m.catalyst_replacement = pe.Param(initialize = 1.25,doc = "catalyst replacement in a year")
        m.material_of_construction_factor = pe.Param(initialize = 1.5,doc = "material of construction factor")
        m.pressure_factor = pe.Param(initialize = 1.7,doc = "pressure factor")
        m.residence_time = pe.Param(initialize = 5.0,doc = "residence time in flash,unit:min")
        m.marshall_swift_index = pe.Param(initialize = 1031.8,doc = "marshall and swift index")
        #Variables 
        m.cooler_cost = pe.Var(m.cooler,within = pe.PositiveReals,doc= 'cost for each cooler',initialize = 1)
        m.utility_requirement_c = pe.Var(m.cooler,domain = pe.PositiveReals,doc = 'unit: 1e12 Kj/yr, for cooler',initialize = 1)
        m.compressor_cost = pe.Var(m.compressor,within = pe.PositiveReals,doc= 'cost for each compressor',initialize = 1)
        m.electricity_requirement = pe.Var(m.compressor,domain = pe.NonNegativeReals,doc = 'unit:kW',initialize = 0) 
        m.furnace_cost = pe.Var(m.pyrolysis_furnance,within = pe.PositiveReals,doc = 'cost for each furnace',initialize = 1)
        m.cost_of_distillation = pe.Var(m.distillation_column,within = pe.PositiveReals,doc = 'cost for each distillation',bounds = (0,1e6),initialize = 1)
        m.heater_cost = pe.Var(m.heater,within = pe.PositiveReals,doc= 'cost for each heater',initialize = 1)
        m.utility_requirement_h = pe.Var(m.heater,domain = pe.Reals,doc = 'unit: 1e12 Kj/yr',initialize = 0, bounds = (0,None))
        m.cost_of_oxychlorination_reactor = pe.Var(m.oxychlorination_reactor,within = pe.PositiveReals,initialize = 1)
        m.cost_of_direct_chlorination_reactor = pe.Var(m.direct_chlorination_reactor,within = pe.PositiveReals,initialize = 1)
        m.cost_of_flash = pe.Var(m.flash,within = pe.PositiveReals,initialize = 1)

        ##stream matches

        self.equipments = ['compressor','cooler','heater','furnace','distillation','direct','oxychlorination','pump','valve','flash']
        self.inlet_streams  = dict()
        self.outlet_streams = dict()
        self.vapor_outlets = dict()
        self.liquid_outlets = dict()
        self.light_key_components = dict()
        self.heavy_key_components = dict()
        self.light_components = dict()
        self.heavy_components = dict()
        self.key_components = dict()
        self.flash_key_components = dict()
        for i in self.equipments:
            self.inlet_streams[i] = dict()
            self.outlet_streams[i] = dict()
        self.vapor_outlets['flash'] = dict()
        self.vapor_outlets['distillation'] = dict()
        self.liquid_outlets['flash'] = dict()
        self.liquid_outlets['distillation'] = dict()
        self.light_key_components['distillation'] = dict()
        self.light_components['distillation'] = dict()
        self.heavy_key_components['distillation'] = dict()
        self.heavy_components['distillation'] = dict()
        self.key_components['distillation'] = dict()
        self.flash_key_components['flash'] = dict()
        #inlet streams 
        self.inlet_streams['compressor'][1] = 12
        self.inlet_streams['compressor'][2] = 13
        self.inlet_streams['cooler'][2] = 15
        self.inlet_streams['compressor'][3] = 16 
        self.inlet_streams['compressor'][4] = 27 
        self.inlet_streams['heater'][2] = 18
        self.inlet_streams['heater'][3] = 20
        self.inlet_streams['oxychlorination'][1] = 21
        self.inlet_streams['cooler'][3] = 23
        self.inlet_streams['flash'][1] = 24
        self.inlet_streams['distillation'][1] = 26
        self.inlet_streams['heater'][1] = 5
        self.inlet_streams['direct'][1] = 8
        self.inlet_streams['cooler'][1] = 9
        self.inlet_streams['compressor'][5] = 33
        self.inlet_streams['compressor'][6] = 34
        self.inlet_streams['cooler'][4] = 36 
        self.inlet_streams['compressor'][7] = 37
        self.inlet_streams['compressor'][8] = 66
        self.inlet_streams['furnace'][1] = 41
        self.inlet_streams['furnace'][2] = 42
        self.inlet_streams['cooler'][5] = 45
        self.inlet_streams['pump'][1] = 46
        self.inlet_streams['pump'][2] = 63
        self.inlet_streams['valve'][1] = 47
        self.inlet_streams['distillation'][2] = 49
        self.inlet_streams['distillation'][3] = 50
        self.inlet_streams['distillation'][4] = 51
        self.inlet_streams['distillation'][5] = 58
        self.inlet_streams['heater'][4] = 64
        self.inlet_streams['heater'][5] = 67

        # outlet streams 
        self.outlet_streams['compressor'][1] = 14
        self.outlet_streams['compressor'][2] = 15
        self.outlet_streams['cooler'][1] = 10
        self.outlet_streams['cooler'][2] = 16
        self.outlet_streams['compressor'][3] = 17 
        self.outlet_streams['compressor'][4] = 29
        self.outlet_streams['heater'][1] = 7
        self.outlet_streams['heater'][2] = 19
        self.outlet_streams['heater'][3] = 21
        self.outlet_streams['heater'][4] = 64
        self.outlet_streams['heater'][5] = 68
        self.outlet_streams['oxychlorination'][1] = 23
        self.outlet_streams['cooler'][3] = 24
        self.outlet_streams['flash'][1] = 7
        self.outlet_streams['direct'][1] = 9
        self.outlet_streams['cooler'][1] = 10
        self.outlet_streams['compressor'][5] = 35
        self.outlet_streams['compressor'][6] = 36
        self.outlet_streams['cooler'][4] = 37 
        self.outlet_streams['compressor'][7] = 38
        self.outlet_streams['compressor'][8] = 67
        self.outlet_streams['furnace'][1] = 43
        self.outlet_streams['furnace'][2] = 44
        self.outlet_streams['cooler'][5] = 46
        self.outlet_streams['pump'][1] = 47
        self.outlet_streams['pump'][2] = 64
        self.outlet_streams['valve'][1] = 48
        #vapor outlets
        self.vapor_outlets['flash'][1] = 25
        self.vapor_outlets['distillation'][1] = 30
        self.vapor_outlets['distillation'][2] = 51
        self.vapor_outlets['distillation'][3] = 57
        self.vapor_outlets['distillation'][4] = 53
        self.vapor_outlets['distillation'][5] = 61
        #liquid outlets 
        self.liquid_outlets['flash'][1] = 27
        self.liquid_outlets['distillation'][1] = 31
        self.liquid_outlets['distillation'][2] = 52
        self.liquid_outlets['distillation'][3] = 58
        self.liquid_outlets['distillation'][4] = 54
        self.liquid_outlets['distillation'][5] = 62
        # light key component
        self.light_key_components['distillation'][1] = 'ethylene di-chloride'
        self.light_key_components['distillation'][2] = 'vinyl chloride'
        self.light_key_components['distillation'][3] = 'hydrogen chloride'
        self.light_key_components['distillation'][4] = 'hydrogen chloride'
        self.light_key_components['distillation'][5] = 'vinyl chloride'
        # light component
        self.light_components['distillation'][1] = 'ethylene di-chloride'
        self.light_components['distillation'][2] = 'hydrogen chloride'
        self.light_components['distillation'][3] = 'hydrogen chloride'
        self.light_components['distillation'][4] = 'hydrogen chloride'
        self.light_components['distillation'][5]  = 'vinyl chloride'
        # heavy key component 
        self.heavy_key_components['distillation'][1]  = 'water'
        self.heavy_key_components['distillation'][2]  = 'ethylene di-chloride'
        self.heavy_key_components['distillation'][3]  = 'vinyl chloride'
        self.heavy_key_components['distillation'][4]  = 'vinyl chloride'
        self.heavy_key_components['distillation'][5] = 'ethylene di-chloride'
        # heavy component 
        self.heavy_components['distillation'][1]  = 'water'
        self.heavy_components['distillation'][2]  = 'ethylene di-chloride'
        self.heavy_components['distillation'][3]  = 'ethylene di-chloride'
        self.heavy_components['distillation'][4]  = 'vinyl chloride'
        self.heavy_components['distillation'][5]  = 'ethylene di-chloride'
        # key component 
        for unit_number in range(1,6):
            self.key_components['distillation'][unit_number] = [self.heavy_key_components['distillation'][unit_number] ,self.light_key_components['distillation'][unit_number]]
        #flash key component
        self.flash_key_components['flash'][1] = 'hydrogen chloride'

        ## parameters with index
        a = {}
        a['vinyl chloride'] = 20750
        m.heat_vapor = pe.Param(m.components,initialize= a ,default = 0)
        b = {}
        b[1] = -182213
        m.heat_of_reaction_d = pe.Param(m.direct_chlorination_reactor,initialize = b )
        c = {}
        c[1] = -239450
        m.heat_of_reaction_o = pe.Param(m.oxychlorination_reactor,initialize = c)
        d = {}
        d[1] = 72760
        d[2] = 72760
        m.heat_of_reaction_p = pe.Param(m.pyrolysis_furnance,initialize = d)
        M = {}
        M[1] = 5.0
        M[2] = 30
        m.upper_bound_pressure = pe.Param(m.pyrolysis_furnance,initialize = M)
        n = {}
        n[1] = 15.90
        n[2] = 17.80
        m.upper_bound_consumption = pe.Param(m.pyrolysis_furnance,initialize = n)
        e = {}
        e[1] = 1.2
        e[2] = 1.5
        m.material_of_construction_factor_f = pe.Param(m.pyrolysis_furnance,initialize = e)
        f = {}
        f[1] = 12
        f[2] = 950
        f[3] = 50
        f[4] = 36
        f[5] = 45
        m.average_flow = pe.Param(m.distillation_column,initialize = f)
        g = {}
        g['ethylene'] = 1.
        m.stream_1_composition = pe.Param(m.components,initialize = g,default = 0)
        h = {}
        h['chlorine']  = 1.
        m.stream_2_composition = pe.Param(m.components,initialize = h,default = 0)
        i = {}
        i['nitrogen'] = 0.79
        i['oxygen'] = 0.21
        m.stream_3_composition = pe.Param(m.components,initialize = i,default = 0)
        j = {}
        j['oxygen'] = 1.0
        m.stream_4_composition = pe.Param(m.components,initialize = j,default = 0)
        self.CPpure = {}
        self.CPpure['nitrogen'] = 25
        self.CPpure['oxygen'] = 25
        self.CPpure['ethylene'] = 50
        self.CPpure['hydrogen chloride'] = 40
        self.CPpure['vinyl chloride'] = 200
        self.CPpure['chlorine'] = 100
        self.CPpure['ethylene di-chloride'] = 400
        self.CPpure['water'] = 100
        k = {}
        k[41] = 0.0006
        k[42] = 0.00072
        m.conversion_corelation = pe.Param(m.stream,initialize = k, default = 0)

        ##parameter set with pandas
        self.Antoine = pd.read_csv('C:/Users/hithi/Desktop/vinyl/antoine_coefficient_1.csv')
        self.antoine = self.Antoine.set_index('name').T.to_dict('list')
        self.gcomp = pd.read_csv('C:/Users/hithi/Desktop/vinyl/Gcomp.csv')
        self.gcomp = self.gcomp.fillna(0)
        self.Gcomp = self.gcomp.set_index('stream').to_dict('list')
        self.Stream_CP = list()
        for stream in m.stream:
            self.Stream_CP.append(sum(self.Gcomp[comp][stream-1] * self.CPpure[comp] for comp in m.components))
        ## pe.Variables     
        m.split_fraction = pe.Var(m.stream, domain = pe.NonNegativeReals,bounds = (0,1),initialize = 0)
        m.split_fraction[27].setub(0)
        m.stream_flowrate = pe.Var(m.stream,domain = pe.NonNegativeReals,bounds = (0,50),initialize = 6)
        m.component_stream_flowrate = pe.Var(m.stream,m.components,domain = pe.NonNegativeReals,bounds = (0,50))
        m.pressure = pe.Var(m.stream,domain = pe.NonNegativeReals,bounds = (0,30),initialize = 2.0) 
        m.pressure[8].setlb(1)
        m.pressure[8].setub(2)
        m.temperature = pe.Var(m.stream,domain = pe.NonNegativeReals,bounds = (0,10)) 
        m.temperature[1].fix(3)
        m.temperature[8].setlb(3.23)
        m.temperature[8].setub(3.38)
        for stream in [21,22]:
            m.temperature[stream].setlb(4.93)
            m.temperature[stream].setub(5.08)
        for stream in [10,12,36,37,38,45,46]:
            m.temperature[stream].setlb(3.0)
        for stream in [30,31]:
            m.temperature[stream].setlb(2.5)
            m.temperature[stream].setub(7.5)
        for stream in [24,25,26]:
            m.temperature[stream].setlb(2.5)
            m.temperature[stream].setub(5.5)
        for stream in [36,37,38,42,44]:
            m.temperature[stream].setub(8.23)
        m.temperature[41].setlb(6.0)
        m.temperature[41].setub(9.5)
        m.temperature[42].setlb(7.73)
        m.temperature[42].setub(8.23)
        m.temperature[44].setlb(7.73)
        m.temperature[44].setub(8.23)
        m.temperature[45].setlb(3.0)
        m.temperature[45].setub(7.8)
        m.temperature[46].setlb(3.0)
        m.temperature[46].setub(7.8)
        for stream in [50,57,58,61,62]:
            m.temperature[stream].setlb(3.0)
            m.temperature[stream].setub(9.5)
        m.temperature[68].setlb(4.93)
        m.temperature[68].setub(5.08)

        for stream in [12,27,29,30,31,50,57,61,62,66,68]:
            m.pressure[stream].setlb(1.0)
        for stream in [21,22,24,25,26,67]:
            m.pressure[stream].setlb(2.5)
            m.pressure[stream].setub(6.5)
        m.pressure[68].setub(6.5)
        m.pressure[30].setub(7.5)
        m.pressure[31].setub(7.5)
        m.pressure[42].setlb(25)
        m.pressure[44].setlb(25)
        m.pressure[36].setub(5)
        m.pressure[36].setub(15)
        m.pressure[37].setub(5)
        m.pressure[37].setub(15)
        m.pressure[38].setub(5)
        for stream in [1,2,5,6,7,9,10,20,21,23,24,32,34,70]:
            m.stream_flowrate[stream].setlb(1.0)
        
        m.vapor_pressure = pe.Var(m.stream,m.components,domain = pe.Reals,bounds = (0,None),initialize = 5) # ?? doc for unit for pe.Variables
        m.split_fraction[5] = 0.5
        m.split_fraction[59] = 0.8
        m.electricity_requirement[1] = 20.7362
        m.electricity_requirement[6] = 111.779
        m.electricity_requirement[7] = 23.6241
        m.utility_requirement_c[1] = 1.27335
        m.utility_requirement_c[3] = 0.31595
        m.utility_requirement_c[4] = 0.646112
        m.utility_requirement_h[1] = 0.00522984
        m.utility_requirement_h[2] = 0.00348001
        m.utility_requirement_h[4] = 1.01767
        m.utility_requirement_h[5] = 0.0702163
        ## initialization with pandas
        self.ini_ = pd.read_csv('C:/Users/hithi/Desktop/vinyl/bounds.csv')
        self.stream_number = self.ini_.iloc[:,[0]].dropna().values
        self.comp_number = self.ini_.iloc[:,[4]].dropna().values
        self.comp_comp = self.ini_.iloc[:,[5]].dropna().values
        self.vp_number = self.ini_.iloc[:,[7]].dropna().values
        self.vp_comp = self.ini_.iloc[:,[8]].dropna().values
        # for i in range(len(self.stream_number)):
        #     m.stream_flowrate[self.stream_number[i,0]].fix(self.ini_.iloc[:,[1]].dropna().values[i,0])
        #     m.pressure[self.stream_number[i,0]].fix(self.ini_.iloc[:,[2]].dropna().values[i,0])
        #     m.temperature[self.stream_number[i,0]].fix(self.ini_.iloc[:,[3]].dropna().values[i,0])
        # for i in range(len(self.comp_number)):
        #     m.component_stream_flowrate[self.comp_number[i,0],self.comp_comp[i,0]].fix(self.ini_.iloc[:,[6]].dropna().values[i,0])
        # for i in range(len(self.vp_number)):
        #     m.vapor_pressure[self.vp_number[i,0],self.vp_comp[i,0]].fix(self.ini_.iloc[:,[9]].dropna().values[i,0])

        for i in range(len(self.stream_number)):
            m.stream_flowrate[self.stream_number[i,0]] = self.ini_.iloc[:,[1]].dropna().values[i,0]
            m.pressure[self.stream_number[i,0]] = self.ini_.iloc[:,[2]].dropna().values[i,0]
            m.temperature[self.stream_number[i,0]] = self.ini_.iloc[:,[3]].dropna().values[i,0]
        for i in range(len(self.comp_number)):
            m.component_stream_flowrate[self.comp_number[i,0],self.comp_comp[i,0]] = self.ini_.iloc[:,[6]].dropna().values[i,0]
        for i in range(len(self.vp_number)):
            m.vapor_pressure[self.vp_number[i,0],self.vp_comp[i,0]] = self.ini_.iloc[:,[9]].dropna().values[i,0]
        ##global constraints
        def component_match(_m,_c):
            return m.stream_flowrate[_c] == sum(m.component_stream_flowrate[_c,d] for d in m.components)
        m.Component_match = pe.Constraint(m.stream,rule = component_match)
        def production_specify(_m):
            return m.component_stream_flowrate[70,'vinyl chloride'] >= 0.90 * m.stream_flowrate[70]
        m.Production_specify = pe.Constraint(rule = production_specify)
        def ethylene_inlet_stream_specify(_m,_c):
            return m.component_stream_flowrate[1,_c] == m.stream_flowrate[1] * m.stream_1_composition[_c]
        m.Ethylene_inlet_stream_specify = pe.Constraint(m.components, rule = ethylene_inlet_stream_specify)
        def chlorine_inlet_stream_specify(_m,_c):
            return m.component_stream_flowrate[2,_c] == m.stream_flowrate[2] * m.stream_2_composition[_c]
        m.chlorine_inlet_stream_specify = pe.Constraint(m.components,rule = chlorine_inlet_stream_specify)
        def oxygen_inlet_stream_specify(_m_,_c):
            return m.component_stream_flowrate[4,_c] == m.stream_flowrate[4] * m.stream_4_composition[_c]
        m.Oxygen_inlet_stream_specify = pe.Constraint(m.components,rule = oxygen_inlet_stream_specify)
        def air_inlet_stream_specify(_m,_c):
            return m.component_stream_flowrate[3,_c] == m.stream_flowrate[3] * m.stream_3_composition[_c]
        m.Air_inlet_stream_specify = pe.Constraint(m.components,rule = air_inlet_stream_specify)
        def stream_join_flowsheet(_m,_c):
            return m.component_stream_flowrate[40,_c] == m.component_stream_flowrate[39,_c] + m.component_stream_flowrate[65,_c] 
        m.Stream_join_flowsheet = pe.Constraint(m.components,rule = stream_join_flowsheet)
        def temp_join_flowsheet(_m):
            return m.temperature[40] == m.temperature[65]
        m.Temp_join_flowsheet_2 = pe.Constraint(rule = temp_join_flowsheet)
        def p_join_flowsheet_2(_m):
            return m.pressure[40] == m.pressure[65]
        m.P_join_flowsheet_2 = pe.Constraint(rule = p_join_flowsheet_2)
        def p_join_flowsheet_1(_m):
            return m.pressure[40] == m.pressure[39]
        m.P_join_flowsheet_1 = pe.Constraint(rule = p_join_flowsheet_1)
        # ************************************************
        # compressor selection ---- outside disjunction 
        # ************************************************
        @m.Disjunct()
        def test_1(disj):
            self.build_stream_doesnt_exist(disj,34)
            self.build_stream_doesnt_exist(disj,36)
            self.build_stream_doesnt_exist(disj,37)
            self.build_stream_doesnt_exist(disj,38)
        @m.Disjunct()
        def test_2(disj):
            self.build_stream_doesnt_exist(disj,33)
            self.build_stream_doesnt_exist(disj,35)

        m.Test = gdp.Disjunction(expr = [m.test_1,m.test_2])
    
        # ************************************
        # Objective
        # ************************************
        def profit(m):
            return -( 11840 * m.stream_flowrate[70] + 1000 * m.stream_flowrate[69]- 6055 * m.stream_flowrate[1]  - 1000 * m.stream_flowrate[71])
            + sum((m.costelec * m.electricity_requirement[i] + m.compressor_cost[i]) for i in m.compressor) + sum(m.cost_of_flash[i] for i in m.flash)+ sum(m.cost_of_distillation[i] for i in m.distillation_column) + sum(m.cost_of_oxychlorination_reactor[i]for i in m.oxychlorination_reactor) + sum(m.cost_of_direct_chlorination_reactor[i] for i in m.direct_chlorination_reactor) + sum(m.furnace_cost[i]for i in m.pyrolysis_furnance) + sum((m.costqc * m.utility_requirement_c[i] + m.cooler_cost[i]) for i in m.components) + sum(m.costqh * m.utility_requirement_h + m.heater_cost[i] for i in m.components)
        m.Obj = pe.Objective(rule = profit ,sense = pe.minimize)
    # ************************************************
    # streams define -- functions 
    # ************************************************
    def build_stream_doesnt_exist(self,block,stream): ## this is just a test,change later
        m = self.model
        b = pe.Block()
        setattr(block, 'stream_doesnt_exist_' + str(stream), b)
        b.zero_flow_con = pe.Constraint(expr=m.stream_flowrate[stream] == 0)
        def _zero_component_flows(_b, _c):
            return m.component_stream_flowrate[stream, _c] == 0
        b.zero_component_flows_con = pe.Constraint(m.components, rule=_zero_component_flows)

        def _fixed_temp_con(_b):
            return m.temperature[stream] == 0
        b.fixed_temp_con = pe.Constraint(rule=_fixed_temp_con)

        def _fixed_pressure_con(_b):
            return m.pressure[stream] == 0
        b.fixed_pressure_con = pe.Constraint(rule=_fixed_pressure_con)

    def build_equal_streams(self,block,in_stream,out_stream):
        m = self.model
        t = m.temperature
        p = m.pressure
        b = pe.Block()
        setattr(block, 'equal_streams_' + str(in_stream) + '_' + str(out_stream), b)
        def _component_balances(_b, _c):
            return m.component_stream_flowrate[out_stream, _c] == m.component_stream_flowrate[in_stream, _c]
        b.component_balances = pe.Constraint(m.components, rule=_component_balances)

        def _temp_con(_b):
            return t[in_stream] == t[out_stream]
        b.temp_con = pe.Constraint(rule=_temp_con)

        def _pressure_con(_b):
            return p[in_stream] == p[out_stream]
        b.pressure_con = pe.Constraint(rule=_pressure_con)

    def build_compressor(self,block,unit_number):
        u = unit_number
        m = self.model
        in_stream = self.inlet_streams['compressor'][u]
        out_stream = self.outlet_streams['compressor'][u]
        b = pe.Block()
        setattr(block, 'compressor_'+str(u), b)
        b.pressure_ratio = pe.Var(domain = pe.Reals,doc = ' outlet to inlet',bounds = (0,None))
        ini_pressure_ratio = [1.43927,0,0,1,0,1.59198,1.12511,1]
        b.pressure_ratio = ini_pressure_ratio[u-1]
        
        def mass_balance(b_,_c,):
            return m.component_stream_flowrate[in_stream,_c] == m.component_stream_flowrate[out_stream,_c]
        b.Mass_balance =  pe.Constraint(m.components,rule = mass_balance)
        def heat_balance(b_):
            return m.temperature[in_stream] * b.pressure_ratio == m.temperature[out_stream]
        b.Heat_balance = pe.Constraint(rule = heat_balance)
        def energy_balance(b_,_c):
            return m.electricity_requirement[u] == self.alpha * (b.pressure_ratio - 1) * (1/self.compeff) * (self.gamma/(self.gamma - 1)) * 100 * m.temperature[in_stream] * m.stream_flowrate[in_stream] / 60
        b.Energy_balance = pe.Constraint(m.components,rule = energy_balance)
        def pressure_ratio_calculation(b_,_c):
            return b.pressure_ratio == (m.pressure[out_stream] / (m.pressure[in_stream] + 1e-10 )) ** ((self.gamma -1)/self.gamma)
        b.Pressure_ratio_calculation = pe.Constraint(m.components ,rule = pressure_ratio_calculation) 
        def compressor_cost(b_):
            return m.compressor_cost[u] == m.marshall_swift_index / 280 * ( 50.5 * (m.electricity_requirement[u] + 0.001 ) ** 0.706 + 1)
        b.Compressor_cost = pe.Constraint(rule = compressor_cost)

    def build_cooler(self,block,unit_number):
        u = unit_number
        m = self.model
        in_stream = self.inlet_streams['cooler'][u]
        outlet_stream = self.outlet_streams['cooler'][u]
        b = pe.Block()
        setattr(block, 'cooler_'+str(u), b)
        ini_cooler_area = [0.363711,1,0.244148,0.726812,0]
        b.cooler_area = pe.Var(domain = pe.Reals,bounds = (0,None))
        b.cooler_area = ini_cooler_area[u-1]
        def mass_balance(b_,_c):
            return m.component_stream_flowrate[in_stream,_c] == m.component_stream_flowrate[outlet_stream,_c] 
        b.Mass_balance = pe.Constraint(m.components,rule = mass_balance)
        def heat_balance(b_):
            return (m.temperature[in_stream] * 100 * self.Stream_CP[in_stream -1] * m.stream_flowrate[in_stream] - m.temperature[outlet_stream] * 100 * self.Stream_CP[outlet_stream -1] * m.stream_flowrate[outlet_stream] ) * 3600. * 8500. * 1.0E-12 / 60. == m.utility_requirement_c[u]
        b.Heat_balance = pe.Constraint(rule = heat_balance)
        def pressure_relation(b_):
            return m.pressure[in_stream] == m.pressure[outlet_stream]
        b.Pressure_relation = pe.Constraint(rule = pressure_relation)
        def area_calculation(b_):
            return m.utility_requirement_c[u] == b.cooler_area * (0.001 + ((m.temperature[in_stream] -3) - ( m.temperature[outlet_stream] -3))/2) #?? -3
        b.Area_calculation = pe.Constraint(rule = area_calculation)
        def temp_relation(b_):
            return m.temperature[in_stream] >= m.temperature[outlet_stream]
        b.temperature_relation = pe.Constraint(rule = temp_relation)
        def cost_of_cooler(b_):
            return m.cooler_cost[u] == m.marshall_swift_index / 280 * (3.17 * ( b.cooler_area + 0.001) ** 0.641 + 0.75) #?? +0.75
        b.Cost_of_cooler = pe.Constraint(rule = cost_of_cooler)
    def build_furnace(self,block,unit_number):
        u = unit_number 
        m = self.model
        in_stream = self.inlet_streams['furnace'][u]
        out_stream = self.outlet_streams['furnace'][u]
        b = pe.Block()
        setattr(block, 'furnace_'+str(u), b)
        b.conversion = pe.Var(within = pe.Reals,bounds = (0.5,0.6),initialize = 0.5616)
        b.consumption = pe.Var(within = pe.Reals,bounds = (0,20),initialize = 17.8341)
        b.furnace_pressure = pe.Var(within = pe.PositiveReals,bounds = (25,30))
        b.furnace_temperature = pe.Var(within = pe.PositiveReals,bounds =(7.730,8.23),initialize = 7.75333)
        b.furnace_volume = pe.Var(domain = pe.PositiveReals,bounds = (0,100),initialize = 36.49)
        def conversion_rate_calculation(b_):
            return b.conversion == m.conversion_corelation[in_stream] * m.temperature[in_stream] * 100
        b.Conversion_rate = pe.Constraint(rule = conversion_rate_calculation)
        def consumption_calculation(b_):
            return b.consumption == b.conversion * m.component_stream_flowrate[in_stream,'ethylene di-chloride']
        b.Consumption_calculation = pe.Constraint(rule = consumption_calculation)
        def pressure_relation(b_):
            return m.pressure[in_stream] == m.pressure[out_stream]
        b.Pressure_relation = pe.Constraint(rule = pressure_relation)
        def temperature_relation(b_):
            return m.temperature[in_stream]  == m.temperature[out_stream]
        b.Temperature_relation = pe.Constraint(rule = temperature_relation)
        def furnace_P_set_up(b_):
            return m.pressure[in_stream] == b.furnace_pressure
        b.Furnace_P_set_up = pe.Constraint(rule = furnace_P_set_up )
        def furnace_T_set_up(b_):
            return m.temperature[in_stream] == b.furnace_temperature
        b.Furnace_T_set_up = pe.Constraint(rule =furnace_T_set_up)
        def vinyl_calculation(b_):
            return m.component_stream_flowrate[out_stream,'vinyl chloride'] == b.consumption
        b.Vinyl_calculation = pe.Constraint(rule = vinyl_calculation)
        def hydrogen_chloride(b_):
            return m.component_stream_flowrate[out_stream,'hydrogen chloride'] == b.consumption
        b.Hydrogen_chloride = pe.Constraint(rule = hydrogen_chloride)
        def ethylene_dichloride(b_):
            return m.component_stream_flowrate[out_stream,'ethylene di-chloride'] == m.component_stream_flowrate[in_stream,'ethylene di-chloride'] - b.consumption
        b.Ethylene_dichloride = pe.Constraint(rule = ethylene_dichloride)
        def pyrolysis_volume(b_):
            return b.furnace_volume == m.upper_bound_consumption[u] * 0.082 * 7.5 * 100 / m.upper_bound_pressure[u]
        b.Pyrolysis_volume = pe.Constraint(rule = pyrolysis_volume)
        def cost_furnace(b_):
            return m.furnace_cost[u] == m.marshall_swift_index / 280 * m.material_of_construction_factor_f[u] * m.material_of_construction_factor * (5.2 * (b.furnace_volume + 0.001) ** 0.6 + 0.75)  #?? 
        b.Cost_furnace = pe.Constraint(rule = cost_furnace)
    def build_distillation(self,block,unit_number):
        u = unit_number
        m = self.model
        in_stream = self.inlet_streams['distillation'][u]
        vapor_outlet = self.vapor_outlets['distillation'][u]
        liquid_outlet = self.liquid_outlets['distillation'][u]
        heavy_component = self.heavy_components['distillation'][u]
        heavy_key_component = self.heavy_key_components['distillation'][u]
        light_component = self.light_components['distillation'][u]
        light_key_component = self.light_key_components['distillation'][u]
        key_component_list = self.key_components['distillation'][u]
        b = pe.Block()
        setattr(block, 'distillation_'+str(u), b)
        b.min_number_of_tray = pe.Var(domain = pe.PositiveReals,bounds = (2,199))
        b.number_of_trays = pe.Var(domain = pe.Reals,bounds = (0,30))
        b.min_reflux_ratio = pe.Var(domain = pe.PositiveReals,bounds = (0.001,30),initialize = 0.001)
        b.reflux_ratio = pe.Var(domain = pe.Reals,bounds = (0,30),initialize = 0)
        b.average_volatility = pe.Var(domain = pe.PositiveReals,bounds = (1.1,100))
        b.cost_of_distillation = pe.Var(domain = pe.PositiveReals)
        b.column_pressure = pe.Var(domain = pe.PositiveReals,bounds = (1,30))
        ini_average_volatility = [6.58773,1.1,50.2248,1.1,12.2081]
        ini_column_pressure = [2,1,4.84493,1,1]
        ini_number_of_trays = [43.7273,0,20.3567,0,38.2632]
        ini_min_number_of_tray = [10.9318,2,5.08917,2,9.5658]
        ini_reflux_ratio = [0.23884,0,0.067786,0,0.190702]
        ini_min_reflux_ratio = [0.199033,0.001,0.0564883,0.001,0.158919]
        b.column_pressure = ini_column_pressure[u-1]
        b.average_volatility = ini_average_volatility[u-1]
        b.number_of_trays = ini_number_of_trays[u-1] 

        b.min_number_of_tray = ini_min_number_of_tray[u-1]
        b.min_reflux_ratio = ini_min_reflux_ratio[u-1]
        b.reflux_ratio = ini_reflux_ratio[u-1]
        def mass_balance(b_,_c):
            return m.component_stream_flowrate[in_stream,_c] ==  m.component_stream_flowrate[vapor_outlet,_c] + m.component_stream_flowrate[liquid_outlet,_c]
        b.Mass_balance = pe.Constraint(m.components,rule = mass_balance)
        def bottom_vapor_pressure_corraltion(b_,_c):
            if _c in key_component_list:
                antonie_A = self.antoine[_c][0]
                antonie_B = self.antoine[_c][1]
                antonie_C = self.antoine[_c][2]
                return m.vapor_pressure[liquid_outlet,_c] * 7500.6168 == exp((antonie_A - antonie_B ) /(m.temperature[liquid_outlet] * 100 + antonie_C))
            else:
                return pe.Constraint.Skip
        b.Bottom_vapor_pressure_corraltion = pe.Constraint(m.components, rule = bottom_vapor_pressure_corraltion)
        def top_vapor_pressure_corraltion(b_,_c):
            if _c in key_component_list:
                antonie_A = self.antoine[_c][0]
                antonie_B = self.antoine[_c][1]
                antonie_C = self.antoine[_c][2]
                return m.vapor_pressure[vapor_outlet,_c] * 7500.6168  == exp((antonie_A - antonie_B) /(m.temperature[vapor_outlet] * 100 + antonie_C))
            else:
                return pe.Constraint.Skip
        b.Top_vapor_pressure_corraltion = pe.Constraint(m.components, rule = top_vapor_pressure_corraltion)
        def average_relative_volatility(b_):
            return 0.5 * (m.vapor_pressure[liquid_outlet,light_key_component] / m.vapor_pressure[liquid_outlet,heavy_key_component]) * (m.vapor_pressure[vapor_outlet,light_key_component] / m.vapor_pressure[vapor_outlet,heavy_key_component]) == b.average_volatility
        b.Average_relative_volatility = pe.Constraint(rule = average_relative_volatility)
        def underwood_equation(b_):
            return m.component_stream_flowrate[in_stream,light_key_component] * b.min_reflux_ratio * (b.average_volatility -1) == m.stream_flowrate[in_stream]
        b.Underwood_equation = pe.Constraint(rule = underwood_equation)
        def actual_reflux_calculation(b_):
            return b.reflux_ratio == 1.2 * b.min_reflux_ratio 
        b.Actual_reflux_calculation = pe.Constraint(rule = actual_reflux_calculation)
        def fenske_equation(b_):
            return b.min_number_of_tray * log(b.average_volatility) == log((m.stream_flowrate[vapor_outlet] + self.eps1) /(m.component_stream_flowrate[vapor_outlet,heavy_key_component] + self.eps1)) * (m.stream_flowrate[liquid_outlet] + self.eps1) /(m.component_stream_flowrate[liquid_outlet,light_key_component] + self.eps1)
        b.Fenske_equation = pe.Constraint(rule = fenske_equation)
        def actual_tray_number(b_):
            return b.number_of_trays == b.min_number_of_tray * 2 / self.disteff
        b.Actual_tray_number = pe.Constraint(rule = actual_tray_number)
        def recovery_specification(b_):
            return m.component_stream_flowrate[vapor_outlet,heavy_key_component] <= 0.05 * m.component_stream_flowrate[in_stream,heavy_key_component]
        b.recovery_specification = pe.Constraint(rule = recovery_specification)
        def heavy_component_calculation(b_):
            return m.component_stream_flowrate[liquid_outlet,heavy_component]  == m.component_stream_flowrate[in_stream,heavy_component]
        b.Heavy_component_calculation = pe.Constraint(rule = heavy_component_calculation)
        def light_component_calculation(b_):
            return  m.component_stream_flowrate[vapor_outlet,light_component]  == m.component_stream_flowrate[in_stream,light_component]
        b.Light_component_calculation = pe.Constraint(rule = light_component_calculation)
        def inlet_pressure_relation(b_):
            return b.column_pressure <= m.pressure[in_stream]
        b.Inlet_pressure_relation = pe.Constraint(rule = inlet_pressure_relation)
        def bottom_vapor_pressure_relation(b_):
            return b.column_pressure ==  m.vapor_pressure[liquid_outlet,heavy_key_component]
        b.Bottom_vapor_pressure_relation = pe.Constraint(rule = bottom_vapor_pressure_relation)
        def top_vapor_pressure_relation(b_):
            if u != 1:
                return b.column_pressure == m.vapor_pressure[vapor_outlet,light_component]
            else:
                return pe.Constraint.Skip
        b.Top_vapor_pressure_relation = pe.Constraint(rule = top_vapor_pressure_relation)
        def liquid_outlet_pressure_relation(b_):
            return b.column_pressure == m.pressure[liquid_outlet]
        b.Liquid_outlet_pressure_relation = pe.Constraint(rule = liquid_outlet_pressure_relation)
        def vapor_outlet_pressure_relation(b_):
            return b.column_pressure == m.pressure[vapor_outlet]
        b.Vapor_outlet_pressure_relation = pe.Constraint(rule = vapor_outlet_pressure_relation)
        def distillation_cost_calc(b_):
            return m.cost_of_distillation[u] == b.number_of_trays  * m.marshall_swift_index / 280 * m.pressure_factor * 1.7 * (0.189 * m.average_flow[u] + 3)
        b.Distillation_cost_calc = pe.Constraint(rule = distillation_cost_calc)
    def build_pump(self,block,unit_number):
        u = unit_number
        m = self.model
        in_stream = self.inlet_streams['pump'][u]
        outlet_stream = self.outlet_streams['pump'][u]
        b = pe.Block()
        setattr(block, 'pump_'+str(u), b)
        def mass_balance(b_,_c):
            return m.component_stream_flowrate[in_stream,_c] == m.component_stream_flowrate[outlet_stream,_c] 
            b.Mass_balance = pe.Constraint(m.components,rule = mass_balance)
        def pressure_relation(b_):
            return m.pressure[in_stream] <= m.pressure[outlet_stream]
        b.Pressure_relation = pe.Constraint(rule = pressure_relation)
        def temp_relation(b_):
            return m.temperature[in_stream] == m.temperature[outlet_stream]
        b.temperature_relation = pe.Constraint(rule = temp_relation)
    def build_valve(self,block,unit_number):
        u = unit_number
        m = self.model
        in_stream = self.inlet_streams['valve'][u]
        outlet_stream = self.outlet_streams['valve'][u]
        b = pe.Block()
        setattr(block, 'valve_'+str(u), b)

        def mass_balance(b_,_c):
            return m.component_stream_flowrate[in_stream,_c] == m.component_stream_flowrate[outlet_stream,_c] 
        b.Mass_balance = pe.Constraint(m.components,rule = mass_balance)
        def pressure_relation(b_):
            return m.pressure[in_stream] >= m.pressure[outlet_stream]
        b.Pressure_relation = pe.Constraint(rule = pressure_relation)
        def temp_relation(b_):
            return m.temperature[in_stream] / (m.pressure[in_stream] ** ((self.gamma-1)/self.gamma)) == m.temperature[outlet_stream] / (m.pressure[outlet_stream] ** ((self.gamma-1)/self.gamma))
        b.temperature_relation = pe.Constraint(rule = temp_relation)
        
    def build_heater(self,block,unit_number):
        u = unit_number
        m = self.model
        in_stream = self.inlet_streams['heater'][u]
        outlet_stream = self.outlet_streams['heater'][u]
        b = pe.Block()
        setattr(block, 'heater_'+str(u), b)
        ini_heater_area = [0.0450848,0.0113322,0,0.567691,0.0726877]
        b.heater_area = pe.Var(domain = pe.Reals,bounds = (0,None))
        b.heater_area = ini_heater_area[u-1]
        def mass_balance(b_,_c):
            return m.component_stream_flowrate[in_stream,_c] == m.component_stream_flowrate[outlet_stream,_c]
        b.Mass_balance = pe.Constraint(m.components,rule = mass_balance)
        def heat_balance(b_):
            return ( m.temperature[outlet_stream] * 100 * self.Stream_CP[outlet_stream - 1] * m.stream_flowrate[outlet_stream] - m.temperature[in_stream] * 100 * self.Stream_CP[in_stream - 1] * m.stream_flowrate[in_stream]) * 3600. * 8500. * 1.0E-12 / 60. == m.utility_requirement_h[u]
        b.Heat_balance = pe.Constraint(rule = heat_balance)
        def pressure_relation(b_):
            return m.pressure[in_stream] == m.pressure[outlet_stream]
        b.Pressure_relation = pe.Constraint(rule = pressure_relation)
        def area_calculation(b_):
            return m.utility_requirement_h[u] == b.heater_area * (0.001 + ((8 - m.temperature[in_stream]) - (8 - m.temperature[outlet_stream]))/2)
        b.Area_calculation = pe.Constraint(rule = area_calculation)
        def temp_relation(b_):
            return m.temperature[in_stream] <= m.temperature[outlet_stream]
        b.temperature_relation = pe.Constraint(rule = temp_relation)
        def cost_of_heater(b_):
            return m.heater_cost[u] == m.marshall_swift_index / 280 * (3.17 * ( b.heater_area + 0.001) ** 0.641 + 0.75)
        b.Cost_of_heater = pe.Constraint(rule = cost_of_heater)

    def build_flash(self,block,unit_number): # ?? careful about binary == 0 
        u = unit_number
        m = self.model
        in_stream = self.inlet_streams['flash'][u]
        vapor_outlet = self.vapor_outlets['flash'][u]
        liquid_outlet = self.liquid_outlets['flash'][u]
        flash_key_component = self.flash_key_components['flash'][u]
        b = pe.Block()
        setattr(block, 'flash_'+str(u), b)
        b.flash_vapor_phase_recovery = pe.Var(m.components,domain = pe.PositiveReals,initialize = 1)
        b.flash_temp = pe.Var(domain = pe.PositiveReals,doc = 'unit: 100k',bounds = (2.5,5.5),initialize = 5.29032)
        b.flash_pressure = pe.Var(domain = pe.PositiveReals,doc = 'unit: maga-pascal',bounds = (2.5,6.5),initialize = 4.84493 )
        def mass_balance(b_,_c):
            return m.component_stream_flowrate[in_stream,_c] == m.component_stream_flowrate[vapor_outlet,_c] + m.component_stream_flowrate[liquid_outlet,_c]
        b.Mass_balance = pe.Constraint(m.components,rule = mass_balance)
        def vapor_pressure_corraltion(b_,_c):
            antonie_A = self.antoine[_c][0]
            antonie_B = self.antoine[_c][1]
            antonie_C = self.antoine[_c][2]
            return m.vapor_pressure[liquid_outlet,_c] * 7500.6168 == exp((antonie_A - antonie_B) /(m.temperature[liquid_outlet] * 100 + antonie_C))
        b.Vapor_pressure_corraltion = pe.Constraint(m.components,rule = vapor_pressure_corraltion)
        def flash_vapor_recovery_relation(b_,_c): # check later ??
            return b.flash_vapor_phase_recovery[flash_key_component]  * (b.flash_vapor_phase_recovery[_c] * m.vapor_pressure[liquid_outlet,flash_key_component] + (1 - b.flash_vapor_phase_recovery[_c]) * m.vapor_pressure[liquid_outlet,_c]) == m.vapor_pressure[liquid_outlet,flash_key_component] * b.flash_vapor_phase_recovery[_c]
        b.Flash_vapor_recovery_relation = pe.Constraint(m.components,rule = flash_vapor_recovery_relation)
        def equilibrium_relation(b_,_c):
            return m.component_stream_flowrate[vapor_outlet,_c] == m.component_stream_flowrate[in_stream,_c] * b.flash_vapor_phase_recovery[_c]
        b.Equilibrium_relation = pe.Constraint(m.components,rule = equilibrium_relation)
        def pressure_relation(b_):
            return b.flash_pressure * m.stream_flowrate[liquid_outlet] == sum(m.vapor_pressure[liquid_outlet,i] * m.component_stream_flowrate[liquid_outlet,i] for i in m.components)
        b.Pressure_relation = pe.Constraint(rule = pressure_relation)
        def inlet_pressure_relation(b_):
            return b.flash_pressure == m.pressure[in_stream]
        b.Inlet_pressure_relation = pe.Constraint(rule = inlet_pressure_relation)
        def liquid_outlet_pressure_relation(b_):
            return b.flash_pressure == m.pressure[liquid_outlet]
        b.Liquid_outlet_pressure_relation = pe.Constraint(rule = liquid_outlet_pressure_relation)
        def vapor_outlet_pressure_relation(b_):
            return b.flash_pressure == m.pressure[vapor_outlet]
        b.Vapor_outlet_pressure_relation = pe.Constraint(rule = vapor_outlet_pressure_relation)
        def inlet_temp_relation(b_):
            return b.flash_temp == m.temperature[in_stream]
        b.Inlet_temp_relation = pe.Constraint(rule = inlet_temp_relation)
        def liquid_outlet_temp_relation(b_):
            return b.flash_temp == m.temperature[liquid_outlet]
        b.Liquid_outlet_temp_relation = pe.Constraint(rule = liquid_outlet_temp_relation)
        def vapor_outlet_temp_relation(b_):
            return b.flash_temp == m.temperature[vapor_outlet]
        b.Vapor_outlet_temp_relation = pe.Constraint(rule = vapor_outlet_temp_relation)
        def cost_flash_calc(b_):
            return m.cost_of_flash[u] == m.marshall_swift_index / 280 * (0.240 * m.residence_time ** 0.738 * ( m.stream_flowrate[in_stream] + 0.01) ** 0.738 + 1.0)
        b.Cost_flash_calc = pe.Constraint(rule = cost_flash_calc)
    def build_direct_chlorination_reactor(self,block,unit_number):
        u = unit_number
        m = self.model
        in_stream = self.inlet_streams['direct'][u]
        outlet_stream = self.outlet_streams['direct'][u]
        b = pe.Block()
        setattr(block, 'direct_chlorination_reactor_'+str(u), b)
        b.react_constant = pe.Var(domain = pe.PositiveReals,initialize = 0.000000007246069)
        b.conversion_of_key_component = pe.Var(domain = pe.PositiveReals,initialize =1)
        b.key_consumption_rate = pe.Var(domain = pe.PositiveReals,bounds = (0,50),initialize = 8.91704)
        b.reactor_density = pe.Var(domain = pe.PositiveReals, doc = 'unit:Kmol/m^3',initialize = 0.236585)
        b.residence_time = pe.Var(domain = pe.PositiveReals, doc = 'unit: second',initialize = 2,bounds = (2,20))
        b.reactor_volume = pe.Var(domain = pe.PositiveReals, doc = 'unit: cubic meter',initialize = 150.762)
        b.heat_removed = pe.Var(domain = pe.PositiveReals,initialize = 828.648)
        def inlet_stream_match(b_):
            return m.component_stream_flowrate[in_stream,'ethylene'] == m.component_stream_flowrate[in_stream,'chlorine']
        b.Inlet_stream_match = pe.Constraint(rule = inlet_stream_match)
        def react_constant_calculation(b_):
            return b.react_constant == 4.3e10 * exp(-14265 / (m.temperature[in_stream] * 100))
        b.React_constant_calculation = pe.Constraint(rule = react_constant_calculation)
        def key_conversion_calculation(b_):
            return b.conversion_of_key_component == 1 - 0.45 * b.react_constant * m.pressure[in_stream] * m.stream_flowrate[in_stream] / 60
        b.Key_conversion_calculation = pe.Constraint(rule = key_conversion_calculation)
        def consumption_rate_calculation(b_):
            return b.key_consumption_rate == b.conversion_of_key_component * m.component_stream_flowrate[in_stream,'ethylene']
        b.Consumption_rate_calculation = pe.Constraint(rule = consumption_rate_calculation)
        def outlet_match_1(b_):
            return b.key_consumption_rate == m.component_stream_flowrate[outlet_stream,'ethylene di-chloride']
        b.Outlet_match_1 = pe.Constraint(rule = outlet_match_1)
        def outlet_match_2(b_,_c):
            if _c != 'ethylene di-chloride':
                return 0 == m.component_stream_flowrate[outlet_stream,_c]
            else:
                return pe.Constraint.Skip
        b.Outlet_match_2 = pe.Constraint(m.components, rule = outlet_match_2)
        def pressure_relation(b_):
            return m.pressure[in_stream] == m.pressure[outlet_stream]
        b.Pressure_relation = pe.Constraint(rule = pressure_relation)
        def temp_relation(b_):
            return m.temperature[in_stream] <= m.temperature[outlet_stream]
        b.Temp_relation = pe.Constraint(rule = temp_relation) 
        def reactor_density_calculation(b_):
            return b.reactor_density * 0.082 * 100 * m.temperature[outlet_stream] == 10 * 0.97 * m.pressure[outlet_stream]
        b.Reactor_density_calculation = pe.Constraint(rule = reactor_density_calculation)
        def residence_time_calculation(b_):
            return 1 - b.conversion_of_key_component == m.component_stream_flowrate[in_stream,'ethylene'] * exp(- b.conversion_of_key_component * b.residence_time * 3 / b.reactor_density ) 
        b.Residence_time_calculation = pe.Constraint(rule = residence_time_calculation)
        def reactor_volume_calculation(b_):
            return b.reactor_volume * b.reactor_density == b.residence_time * m.stream_flowrate[in_stream]
        b.Reactor_volume_calculation = pe.Constraint(rule = reactor_volume_calculation)
        def energy_balance(b_):
            return b.heat_removed == abs(m.heat_of_reaction_d[u]) * b.key_consumption_rate * 8500 * 60 * 1e-9
        b.Energy_balance = pe.Constraint(rule = energy_balance)
        def temp_calculation(b_):
            return m.temperature[in_stream] + b.heat_removed * 1e+9 / (3 * self.Stream_CP[8] * 2 * 60 * 100 * 8500) == m.temperature[in_stream] 
        b.Temp_calculation = pe.Constraint(rule = temp_calculation)
        def cost_of_direct_calc(b_):
            return m.cost_of_direct_chlorination_reactor[u] == m.marshall_swift_index / 280 * m.material_of_construction_factor * m.pressure_factor * (0.225 * (b.reactor_volume + 0.001) +1) + m.catalyst_replacement * 4 * b.reactor_volume
        b.Cost_of_direct_calc = pe.Constraint(rule = cost_of_direct_calc)
    def build_oxychlorination_reactor(self,block,unit_number):
        u = unit_number
        m = self.model
        in_stream = self.inlet_streams['oxychlorination'][u]
        outlet_stream = self.outlet_streams['oxychlorination'][u]
        b = pe.Block()
        setattr(block, 'oxychlorination_reactor_'+str(u), b)
        b.react_constant = pe.Var(domain = pe.PositiveReals,bounds = (0.261494,0.647941),initialize =0.261494)
        b.conversion_of_key_component = pe.Var(domain = pe.PositiveReals,initialize = 1)
        b.key_consumption_rate = pe.Var(domain = pe.PositiveReals,bounds = (0.950,100),initialize = 8.91704)
        b.reactor_volume = pe.Var(domain = pe.PositiveReals, doc = 'unit: cubic meter',initialize = 146.239)
        b.heat_removed = pe.Var(domain = pe.PositiveReals,initialize = 1088.94)
        def inlet_stream_match_1(b_):
            return m.component_stream_flowrate[in_stream,'ethylene'] == 0.5 * m.component_stream_flowrate[in_stream,'hydrogen chloride']
        b.Inlet_stream_match_1 = pe.Constraint(rule = inlet_stream_match_1)
        def inlet_stream_match_2(b_):
            return m.component_stream_flowrate[in_stream,'ethylene'] == 2.0 * m.component_stream_flowrate[in_stream,'oxygen']
        b.Inlet_stream_match_2 = pe.Constraint(rule = inlet_stream_match_2)
        def react_constant_calculation(b_):
            return b.react_constant == 5.8e+12 * exp( -15150 / (m.temperature[in_stream] * 100))
        b.React_constant_calculation = pe.Constraint(rule = react_constant_calculation)   
        def key_conversion_calculation(b_):
            return b.conversion_of_key_component >= 1 - 0.02 / exp(b.react_constant) 
        b.Key_conversion_calculation = pe.Constraint(rule = key_conversion_calculation)
        def consumption_rate_calculation(b_):
            return b.key_consumption_rate == b.conversion_of_key_component * m.component_stream_flowrate[in_stream,'ethylene']
        b.Consumption_rate_calculation = pe.Constraint(rule = consumption_rate_calculation)
        def outlet_match_1(b_):
            return b.key_consumption_rate == m.component_stream_flowrate[outlet_stream,'ethylene']
        b.Outlet_match_1 = pe.Constraint(rule = outlet_match_1)
        def outlet_match_2(b_):
            return m.component_stream_flowrate[outlet_stream,'ethylene di-chloride'] == m.component_stream_flowrate[in_stream,'ethylene di-chloride'] - b.key_consumption_rate
        b.Outlet_match_2 = pe.Constraint(rule = outlet_match_2)
        def outlet_match_3(b_):
            return m.component_stream_flowrate[outlet_stream,'hydrogen chloride'] == m.component_stream_flowrate[in_stream,'hydrogen chloride'] - 2.0 * b.key_consumption_rate
        b.Outlet_match_3 = pe.Constraint(rule = outlet_match_3)
        def outlet_match_4(b_):
            return m.component_stream_flowrate[outlet_stream,'oxygen'] == m.component_stream_flowrate[in_stream,'oxygen'] - 0.5 * b.key_consumption_rate
        b.Outlet_match_4 = pe.Constraint(rule = outlet_match_4)
        def pressure_relation(b_):
            return m.pressure[in_stream] == m.pressure[outlet_stream]
        b.Pressure_relation = pe.Constraint(rule = pressure_relation)
        def reactor_volume_calculation(b_):
            return b.reactor_volume == b.key_consumption_rate * 0.082 * 5.0 * 100 / 2.5
        b.Reactor_volume_calculation = pe.Constraint(rule = reactor_volume_calculation)
        def energy_balance(b_):
            return b.heat_removed == abs(m.heat_of_reaction_o[u]) * b.key_consumption_rate * 8500 * 60 * 1e-9
        b.Energy_balance = pe.Constraint(rule = energy_balance)
        def temp_relation(b_):
            return m.temperature[in_stream] + b.heat_removed * 1e+9 / (2 * self.Stream_CP[outlet_stream -1 ]  * 60 * 100 * 8500) == m.temperature[outlet_stream]
        b.Temp_relation = pe.Constraint(rule = temp_relation)
        def cost_oxy_calc(b_):
            return m.cost_of_oxychlorination_reactor[u] == m.marshall_swift_index / 280 * m.material_of_construction_factor * m.pressure_factor * (0.125 * (b.reactor_volume + 0.001) ** 0.738 +1) + m.catalyst_replacement * 4 * b.reactor_volume
        b.Cost_oxy_calc = pe.Constraint(rule = cost_oxy_calc)  

def infeasible_constraints(m):
    '''
    This function checks infeasible constraint in the model 
    '''
    log_infeasible_constraints(m) 

def solve_with_minlp(m):
    pe.TransformationFactory('gdp.bigm').apply_to(m, bigM=50)
    # TransformationFactory('gdp.chull').apply_to(m)
    result = pe.SolverFactory('gams').solve(
        m, solver='baron', tee=True, keepfiles = True,tmpdir = 'C:/Users/hithi/Desktop/vinyl/',symbolic_solver_labels=True
        # add_options=[
        #     'GAMS_MODEL.optfile = 1;' 
        # ]
    );
    
    return m      
if __name__ == "__main__":
    m = VinylChlorideModel().model
    # res = solve_with_gdp_opt(m)
    # fix_disjuncts(m)
    # res = fix_optimal(m)
    res = solve_with_minlp(m)
    # for stream in m.stream:
    #     print('stream',stream,'has lb',value(m.temperature[stream].lb),'and ub',value(m.temperature[stream].ub))
    infeasible_constraints(m)
    m.test_1.indicator_var.pprint()
    m.test_2.indicator_var.pprint()
    m.test_2.pprint()