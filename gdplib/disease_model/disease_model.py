#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

# ============================================
# SIR disease model using a low/high transmission parameter
# This is formulated as a disjunctive program
#
# Daniel Word, November 1, 2010
# ============================================

# import packages
from pyomo.environ import *
from pyomo.gdp import *
import math


def build_model():
    """
    Builds the model for the SIR disease model using a low/high transmission parameter.

    The model simulates the spread of an infectious disease over a series of bi-weekly periods, using a disjunctive programming approach to account for variations in disease transmission rates.

    Parameters
    ----------
    None

    Returns
    -------
    model : Pyomo.ConcreteModel
        Pyomo model object that represent the SIR disease model using a low/high transmission parameter.
    """
    # import data
    # Population Data

    pop = [
        15.881351,
        15.881339,
        15.881320,
        15.881294,
        15.881261,
        15.881223,
        15.881180,
        15.881132,
        15.881079,
        15.881022,
        15.880961,
        15.880898,
        15.880832,
        15.880764,
        15.880695,
        15.880624,
        15.880553,
        15.880480,
        15.880409,
        15.880340,
        15.880270,
        15.880203,
        15.880138,
        15.880076,
        15.880016,
        15.879960,
        15.879907,
        15.879852,
        15.879799,
        15.879746,
        15.879693,
        15.879638,
        15.879585,
        15.879531,
        15.879477,
        15.879423,
        15.879370,
        15.879315,
        15.879262,
        15.879209,
        15.879155,
        15.879101,
        15.879048,
        15.878994,
        15.878940,
        15.878886,
        15.878833,
        15.878778,
        15.878725,
        15.878672,
        15.878618,
        15.878564,
        15.878510,
        15.878457,
        15.878402,
        15.878349,
        15.878295,
        15.878242,
        15.878187,
        15.878134,
        15.878081,
        15.878026,
        15.877973,
        15.877919,
        15.877864,
        15.877811,
        15.877758,
        15.877704,
        15.877650,
        15.877596,
        15.877543,
        15.877488,
        15.877435,
        15.877381,
        15.877326,
        15.877273,
        15.877220,
        15.877166,
        15.877111,
        15.877058,
        15.877005,
        15.876950,
        15.876896,
        15.876843,
        15.876789,
        15.876735,
        15.876681,
        15.876628,
        15.876573,
        15.876520,
        15.876466,
        15.876411,
        15.876358,
        15.876304,
        15.876251,
        15.876196,
        15.876143,
        15.876089,
        15.876034,
        15.875981,
        15.875927,
        15.875872,
        15.875819,
        15.875765,
        15.875712,
        15.875657,
        15.875604,
        15.875550,
        15.875495,
        15.875442,
        15.875388,
        15.875335,
        15.875280,
        15.875226,
        15.875173,
        15.875118,
        15.875064,
        15.875011,
        15.874956,
        15.874902,
        15.874849,
        15.874795,
        15.874740,
        15.874687,
        15.874633,
        15.874578,
        15.874525,
        15.874471,
        15.874416,
        15.874363,
        15.874309,
        15.874256,
        15.874201,
        15.874147,
        15.874094,
        15.874039,
        15.873985,
        15.873931,
        15.873878,
        15.873823,
        15.873769,
        15.873716,
        15.873661,
        15.873607,
        15.873554,
        15.873499,
        15.873445,
        15.873391,
        15.873338,
        15.873283,
        15.873229,
        15.873175,
        15.873121,
        15.873067,
        15.873013,
        15.872960,
        15.872905,
        15.872851,
        15.872797,
        15.872742,
        15.872689,
        15.872635,
        15.872580,
        15.872526,
        15.872473,
        15.872419,
        15.872364,
        15.872310,
        15.872256,
        15.872202,
        15.872148,
        15.872094,
        15.872039,
        15.871985,
        15.871932,
        15.871878,
        15.871823,
        15.871769,
        15.871715,
        15.871660,
        15.871607,
        15.871553,
        15.871499,
        15.871444,
        15.871390,
        15.871337,
        15.871282,
        15.871228,
        15.871174,
        15.871119,
        15.871065,
        15.871012,
        15.870958,
        15.870903,
        15.870849,
        15.870795,
        15.870740,
        15.870686,
        15.870633,
        15.870577,
        15.870524,
        15.870470,
        15.870416,
        15.870361,
        15.870307,
        15.870253,
        15.870198,
        15.870144,
        15.870091,
        15.870037,
        15.869982,
        15.869928,
        15.869874,
        15.869819,
        15.869765,
        15.869711,
        15.869656,
        15.869602,
        15.869548,
        15.869495,
        15.869439,
        15.869386,
        15.869332,
        15.869277,
        15.869223,
        15.869169,
        15.869114,
        15.869060,
        15.869006,
        15.868952,
        15.868897,
        15.868843,
        15.868789,
        15.868734,
        15.868679,
        15.868618,
        15.868556,
        15.868489,
        15.868421,
        15.868351,
        15.868280,
        15.868208,
        15.868134,
        15.868063,
        15.867991,
        15.867921,
        15.867852,
        15.867785,
        15.867721,
        15.867659,
        15.867601,
        15.867549,
        15.867499,
        15.867455,
        15.867416,
        15.867383,
        15.867357,
        15.867338,
        15.867327,
        15.867321,
        15.867327,
        15.867338,
        15.867359,
        15.867386,
        15.867419,
        15.867459,
        15.867505,
        15.867555,
        15.867610,
        15.867671,
        15.867734,
        15.867801,
        15.867869,
        15.867941,
        15.868012,
        15.868087,
        15.868161,
        15.868236,
        15.868310,
        15.868384,
        15.868457,
        15.868527,
        15.868595,
        15.868661,
        15.868722,
        15.868780,
        15.868837,
        15.868892,
        15.868948,
        15.869005,
        15.869061,
        15.869116,
        15.869173,
        15.869229,
        15.869284,
        15.869341,
        15.869397,
        15.869452,
        15.869509,
        15.869565,
        15.869620,
        15.869677,
        15.869733,
        15.869788,
        15.869845,
        15.869901,
        15.869956,
        15.870012,
        15.870069,
        15.870124,
        15.870180,
        15.870237,
        15.870292,
        15.870348,
        15.870405,
        15.870461,
        15.870516,
        15.870572,
        15.870629,
        15.870684,
        15.870740,
        15.870796,
        15.870851,
        15.870908,
        15.870964,
        15.871019,
        15.871076,
        15.871132,
        15.871187,
        15.871243,
        15.871300,
        15.871355,
        15.871411,
        15.871467,
        15.871522,
        15.871579,
        15.871635,
        15.871691,
        15.871746,
        15.871802,
        15.871859,
        15.871914,
        15.871970,
        15.872026,
        15.872081,
        15.872138,
        15.872194,
        15.872249,
        15.872305,
        15.872361,
        15.872416,
        15.872473,
        15.872529,
        15.872584,
        15.872640,
        15.872696,
        15.872751,
        15.872807,
        15.872864,
        15.872919,
        15.872975,
        15.873031,
        15.873087,
        15.873142,
        15.873198,
        15.873255,
        15.873310,
        15.873366,
        15.873422,
        15.873477,
        15.873533,
        15.873589,
        15.873644,
        15.873700,
        15.873757,
        15.873811,
        15.873868,
        15.873924,
        15.873979,
        15.874035,
        15.874091,
        15.874146,
        15.874202,
        15.874258,
        15.874313,
        15.874369,
        15.874425,
        15.874481,
        15.874536,
        15.874592,
    ]

    logIstar = [
        7.943245,
        8.269994,
        8.517212,
        8.814208,
        9.151740,
        9.478472,
        9.559847,
        9.664087,
        9.735378,
        9.852583,
        9.692265,
        9.498807,
        9.097634,
        8.388878,
        7.870516,
        7.012956,
        6.484941,
        5.825368,
        5.346815,
        5.548361,
        5.706732,
        5.712617,
        5.709714,
        5.696888,
        5.530087,
        5.826563,
        6.643563,
        7.004292,
        7.044663,
        7.190259,
        7.335926,
        7.516861,
        7.831779,
        8.188895,
        8.450204,
        8.801436,
        8.818379,
        8.787658,
        8.601685,
        8.258338,
        7.943364,
        7.425585,
        7.062834,
        6.658307,
        6.339600,
        6.526984,
        6.679178,
        6.988758,
        7.367331,
        7.746694,
        8.260558,
        8.676522,
        9.235582,
        9.607778,
        9.841917,
        10.081571,
        10.216090,
        10.350366,
        10.289668,
        10.248842,
        10.039504,
        9.846343,
        9.510392,
        9.190923,
        8.662465,
        7.743221,
        7.128458,
        5.967898,
        5.373883,
        5.097497,
        4.836570,
        5.203345,
        5.544798,
        5.443047,
        5.181152,
        5.508669,
        6.144130,
        6.413744,
        6.610423,
        6.748885,
        6.729511,
        6.789841,
        6.941034,
        7.093516,
        7.307039,
        7.541077,
        7.644803,
        7.769145,
        7.760187,
        7.708017,
        7.656795,
        7.664983,
        7.483828,
        6.887324,
        6.551093,
        6.457449,
        6.346064,
        6.486300,
        6.612378,
        6.778753,
        6.909477,
        7.360570,
        8.150303,
        8.549044,
        8.897572,
        9.239323,
        9.538751,
        9.876531,
        10.260911,
        10.613536,
        10.621510,
        10.661115,
        10.392899,
        10.065536,
        9.920090,
        9.933097,
        9.561691,
        8.807713,
        8.263463,
        7.252184,
        6.669083,
        5.877763,
        5.331878,
        5.356563,
        5.328469,
        5.631146,
        6.027497,
        6.250717,
        6.453919,
        6.718444,
        7.071636,
        7.348905,
        7.531528,
        7.798226,
        8.197941,
        8.578809,
        8.722964,
        8.901152,
        8.904370,
        8.889865,
        8.881902,
        8.958903,
        8.721281,
        8.211509,
        7.810624,
        7.164607,
        6.733688,
        6.268503,
        5.905983,
        5.900432,
        5.846547,
        6.245427,
        6.786271,
        7.088480,
        7.474295,
        7.650063,
        7.636703,
        7.830990,
        8.231516,
        8.584816,
        8.886908,
        9.225216,
        9.472778,
        9.765505,
        9.928623,
        10.153033,
        10.048574,
        9.892620,
        9.538818,
        8.896100,
        8.437584,
        7.819738,
        7.362598,
        6.505880,
        5.914972,
        6.264584,
        6.555019,
        6.589319,
        6.552029,
        6.809771,
        7.187616,
        7.513918,
        8.017712,
        8.224957,
        8.084474,
        8.079148,
        8.180991,
        8.274269,
        8.413748,
        8.559599,
        8.756090,
        9.017927,
        9.032720,
        9.047983,
        8.826873,
        8.366489,
        8.011876,
        7.500830,
        7.140406,
        6.812626,
        6.538719,
        6.552218,
        6.540129,
        6.659927,
        6.728530,
        7.179692,
        7.989210,
        8.399173,
        8.781128,
        9.122303,
        9.396378,
        9.698512,
        9.990104,
        10.276543,
        10.357284,
        10.465869,
        10.253833,
        10.018503,
        9.738407,
        9.484367,
        9.087025,
        8.526409,
        8.041126,
        7.147168,
        6.626706,
        6.209446,
        5.867231,
        5.697439,
        5.536769,
        5.421413,
        5.238297,
        5.470136,
        5.863007,
        6.183083,
        6.603569,
        6.906278,
        7.092324,
        7.326612,
        7.576052,
        7.823430,
        7.922775,
        8.041677,
        8.063403,
        8.073229,
        8.099726,
        8.168522,
        8.099041,
        8.011404,
        7.753147,
        6.945211,
        6.524244,
        6.557723,
        6.497742,
        6.256247,
        5.988794,
        6.268093,
        6.583316,
        7.106842,
        8.053929,
        8.508237,
        8.938915,
        9.311863,
        9.619753,
        9.931745,
        10.182361,
        10.420978,
        10.390829,
        10.389230,
        10.079342,
        9.741479,
        9.444561,
        9.237448,
        8.777687,
        7.976436,
        7.451502,
        6.742856,
        6.271545,
        5.782289,
        5.403089,
        5.341954,
        5.243509,
        5.522993,
        5.897001,
        6.047042,
        6.100738,
        6.361727,
        6.849562,
        7.112544,
        7.185346,
        7.309412,
        7.423746,
        7.532142,
        7.510318,
        7.480175,
        7.726362,
        8.061117,
        8.127072,
        8.206166,
        8.029634,
        7.592953,
        7.304869,
        7.005394,
        6.750019,
        6.461377,
        6.226432,
        6.287047,
        6.306452,
        6.783694,
        7.450957,
        7.861692,
        8.441530,
        8.739626,
        8.921994,
        9.168961,
        9.428077,
        9.711664,
        10.032714,
        10.349937,
        10.483985,
        10.647475,
        10.574038,
        10.522431,
        10.192246,
        9.756246,
        9.342511,
        8.872072,
        8.414189,
        7.606582,
        7.084701,
        6.149903,
        5.517257,
        5.839429,
        6.098090,
        6.268935,
        6.475965,
        6.560543,
        6.598942,
        6.693938,
        6.802531,
        6.934345,
        7.078370,
        7.267736,
        7.569640,
        7.872204,
        8.083603,
        8.331226,
        8.527144,
        8.773523,
        8.836599,
        8.894303,
        8.808326,
        8.641717,
        8.397901,
        7.849034,
        7.482899,
        7.050252,
        6.714103,
        6.900603,
        7.050765,
        7.322905,
        7.637986,
        8.024340,
        8.614505,
        8.933591,
        9.244008,
        9.427410,
        9.401385,
        9.457744,
        9.585068,
        9.699673,
        9.785478,
        9.884559,
        9.769732,
        9.655075,
        9.423071,
        9.210198,
        8.786654,
        8.061787,
        7.560976,
        6.855829,
        6.390707,
        5.904006,
        5.526631,
        5.712303,
        5.867027,
        5.768367,
        5.523352,
        5.909118,
        6.745543,
        6.859218,
    ]

    deltaS = [
        9916.490263,
        12014.263380,
        13019.275755,
        12296.373612,
        8870.995603,
        1797.354574,
        -6392.880771,
        -16150.825387,
        -27083.245106,
        -40130.421462,
        -50377.169958,
        -57787.717468,
        -60797.223427,
        -59274.041897,
        -55970.213230,
        -51154.650927,
        -45877.841034,
        -40278.553775,
        -34543.967175,
        -28849.633641,
        -23192.776605,
        -17531.130740,
        -11862.021829,
        -6182.456792,
        -450.481090,
        5201.184400,
        10450.773882,
        15373.018272,
        20255.699431,
        24964.431669,
        29470.745887,
        33678.079947,
        37209.808930,
        39664.432393,
        41046.735479,
        40462.982011,
        39765.070209,
        39270.815830,
        39888.077002,
        42087.276604,
        45332.012929,
        49719.128772,
        54622.190928,
        59919.718626,
        65436.341097,
        70842.911460,
        76143.747430,
        81162.358574,
        85688.102884,
        89488.917734,
        91740.108470,
        91998.787916,
        87875.986012,
        79123.877908,
        66435.611045,
        48639.250610,
        27380.282817,
        2166.538464,
        -21236.428084,
        -43490.803535,
        -60436.624080,
        -73378.401966,
        -80946.278268,
        -84831.969493,
        -84696.627286,
        -81085.365407,
        -76410.847049,
        -70874.415387,
        -65156.276464,
        -59379.086883,
        -53557.267619,
        -47784.164830,
        -42078.001172,
        -36340.061427,
        -30541.788202,
        -24805.281435,
        -19280.817165,
        -13893.690606,
        -8444.172221,
        -3098.160839,
        2270.908649,
        7594.679295,
        12780.079247,
        17801.722109,
        22543.091206,
        26897.369814,
        31051.285734,
        34933.809557,
        38842.402859,
        42875.230152,
        47024.395356,
        51161.516122,
        55657.298307,
        60958.155424,
        66545.635029,
        72202.930397,
        77934.761905,
        83588.207792,
        89160.874522,
        94606.115027,
        99935.754968,
        104701.404975,
        107581.670606,
        108768.440311,
        107905.700480,
        104062.148863,
        96620.281684,
        83588.443029,
        61415.088182,
        27124.031692,
        -7537.285321,
        -43900.451653,
        -70274.062783,
        -87573.481475,
        -101712.148408,
        -116135.719087,
        -124187.225446,
        -124725.278371,
        -122458.145590,
        -117719.918256,
        -112352.138605,
        -106546.806030,
        -100583.803012,
        -94618.253238,
        -88639.090897,
        -82725.009842,
        -76938.910669,
        -71248.957807,
        -65668.352795,
        -60272.761991,
        -55179.538428,
        -50456.021161,
        -46037.728058,
        -42183.912670,
        -39522.184006,
        -38541.255303,
        -38383.665728,
        -39423.998130,
        -40489.466130,
        -41450.406768,
        -42355.156592,
        -43837.562085,
        -43677.262972,
        -41067.896944,
        -37238.628465,
        -32230.392026,
        -26762.766062,
        -20975.163308,
        -15019.218554,
        -9053.105545,
        -3059.663132,
        2772.399618,
        8242.538397,
        13407.752291,
        18016.047539,
        22292.125752,
        26616.583347,
        30502.564253,
        33153.890890,
        34216.684448,
        33394.220786,
        29657.417791,
        23064.375405,
        12040.831532,
        -2084.921068,
        -21390.235970,
        -38176.615985,
        -51647.714482,
        -59242.564959,
        -60263.150854,
        -58599.245165,
        -54804.972560,
        -50092.112608,
        -44465.812552,
        -38533.096297,
        -32747.104307,
        -27130.082610,
        -21529.632955,
        -15894.611939,
        -10457.566933,
        -5429.042583,
        -903.757828,
        2481.947589,
        5173.789976,
        8358.768202,
        11565.584635,
        14431.147931,
        16951.619820,
        18888.807708,
        20120.884465,
        20222.141242,
        18423.168124,
        16498.668271,
        14442.624242,
        14070.038273,
        16211.370808,
        19639.815904,
        24280.360465,
        29475.380079,
        35030.793540,
        40812.325095,
        46593.082382,
        52390.906885,
        58109.310860,
        63780.896094,
        68984.456561,
        72559.442320,
        74645.487900,
        74695.219755,
        72098.143876,
        66609.929889,
        56864.971296,
        41589.295266,
        19057.032104,
        -5951.329863,
        -34608.796853,
        -56603.801584,
        -72678.838057,
        -83297.070856,
        -90127.593511,
        -92656.040614,
        -91394.995510,
        -88192.056842,
        -83148.833075,
        -77582.587173,
        -71750.440823,
        -65765.369857,
        -59716.101820,
        -53613.430067,
        -47473.832358,
        -41287.031890,
        -35139.919259,
        -29097.671507,
        -23178.836760,
        -17486.807388,
        -12046.775779,
        -6802.483422,
        -1867.556171,
        2644.380534,
        6615.829501,
        10332.557518,
        13706.737038,
        17017.991307,
        20303.136670,
        23507.386461,
        26482.194102,
        29698.585356,
        33196.305757,
        37385.914179,
        42872.996212,
        48725.617879,
        54564.488527,
        60453.841604,
        66495.146265,
        72668.620416,
        78723.644870,
        84593.136677,
        89974.936239,
        93439.798630,
        95101.207834,
        94028.126381,
        89507.925620,
        80989.846001,
        66944.274744,
        47016.422041,
        19932.783790,
        -6198.433172,
        -32320.379400,
        -49822.852084,
        -60517.553414,
        -66860.548269,
        -70849.714105,
        -71058.721556,
        -67691.947812,
        -63130.703822,
        -57687.607311,
        -51916.952488,
        -45932.054982,
        -39834.909941,
        -33714.535713,
        -27564.443333,
        -21465.186188,
        -15469.326408,
        -9522.358787,
        -3588.742161,
        2221.802073,
        7758.244339,
        13020.269708,
        18198.562827,
        23211.338588,
        28051.699645,
        32708.577247,
        37413.795242,
        42181.401920,
        46462.499633,
        49849.582315,
        53026.578940,
        55930.600705,
        59432.642178,
        64027.356857,
        69126.843653,
        74620.328837,
        80372.056070,
        86348.152766,
        92468.907239,
        98568.998246,
        104669.511588,
        110445.790143,
        115394.348973,
        119477.553152,
        121528.574511,
        121973.674087,
        121048.017786,
        118021.473181,
        112151.993711,
        102195.999157,
        85972.731130,
        61224.719621,
        31949.279603,
        -3726.022971,
        -36485.298619,
        -67336.469799,
        -87799.366129,
        -98865.713558,
        -104103.651120,
        -105068.402300,
        -103415.820781,
        -99261.356633,
        -94281.850081,
        -88568.701325,
        -82625.711921,
        -76766.776770,
        -70998.803524,
        -65303.404499,
        -59719.198305,
        -54182.230439,
        -48662.904657,
        -43206.731668,
        -37732.701095,
        -32375.478519,
        -27167.508567,
        -22197.211891,
        -17722.869502,
        -13925.135219,
        -10737.893027,
        -8455.327914,
        -7067.008358,
        -7086.991191,
        -7527.693561,
        -8378.025732,
        -8629.383998,
        -7854.586079,
        -5853.040657,
        -1973.225485,
        2699.850783,
        8006.098287,
        13651.734934,
        19139.318072,
        24476.645420,
        29463.480336,
        33899.078820,
        37364.528796,
        38380.214949,
        37326.585649,
        33428.470616,
        27441.000494,
        21761.126583,
        15368.408081,
        7224.234078,
        -2702.217396,
        -14109.682505,
        -27390.915614,
        -38569.562393,
        -47875.155339,
        -53969.121872,
        -57703.473001,
        -57993.198171,
        -54908.391840,
        -50568.410328,
        -45247.622563,
        -39563.224328,
        -33637.786521,
        -27585.345413,
        -21572.074797,
        -15597.363909,
        -9577.429076,
        -3475.770622,
        2520.378408,
        8046.881775,
        13482.345595,
    ]

    beta_set = [
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        26,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        26,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        26,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        26,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        26,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        26,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        26,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        26,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        26,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        26,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        26,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        26,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        26,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        26,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        26,
    ]

    # from new_data_set import * # Uncomment this line to use new data set

    # declare model name
    model = ConcreteModel('SIR Disease Model')

    # declare constants
    bpy = 26  # biweeks per year
    years = 15  # years of data
    bigM = 50.0  # big M for disjunction constraints

    # declare sets
    model.S_meas = RangeSet(1, bpy * years)
    model.S_meas_small = RangeSet(1, bpy * years - 1)
    model.S_beta = RangeSet(1, bpy)

    # define variable bounds
    def _gt_zero(m, i):
        """
        Defines boundary constraints ensuring variables remain greater than zero.

        Parameters
        ----------
        m : Pyomo.ConcreteModel
            SIR disease model using a low/high transmission parameter
        i : int
            index of biweekly periods in the data set

        Returns
        -------
        tuple
            A tuple representing the lower and upper bounds for the variable, ensuring it remains positive.
        """
        return (0.0, 1e7)

    def _beta_bounds(m):
        """
        Sets the bounds for the transmission parameter beta within the model.

        Parameters
        ----------
        m : Pyomo.ConcreteModel
            SIR disease model using a low/high transmission parameter

        Returns
        -------
        tuple
            A tuple representing the lower and upper bounds for the beta variable.
        """
        return (None, 5.0)

    # Define variables
    # Log of estimated cases; All the variables are represented as common logarithm which the log base is 10.
    # The original code inside build_model employs a disjunctive approach with integrated constraints.
    # On the other hand, the commented code uses separate constraints for each scenario, applying the Big-M Reformulation.
    # Binary variables (model.y) are defined inside on the build_model() function. The disjuncts for the Big-M Reformulation is written outside of the code.

    # model.logI = Var(model.S_meas, bounds=_gt_zero, doc='log of estimated cases')
    model.logI = Var(model.S_meas, bounds=(0.001, 1e7), doc='log of estimated cases')
    # log of transmission parameter beta
    # model.logbeta = Var(model.S_beta, bounds=_gt_zero, doc='log of transmission parameter beta')
    model.logbeta = Var(
        model.S_beta, bounds=(0.0001, 5), doc='log of transmission parameter beta'
    )
    # binary variable y over all betas
    # model.y = Var(model.S_beta, within=Binary, doc='binary variable y over all betas')
    # low value of beta
    # model.logbeta_low = Var(bounds=_beta_bounds, doc='low value of beta')
    model.logbeta_low = Var(bounds=(0.0001, 5))
    # high value of beta
    # model.logbeta_high = Var(bounds=_beta_bounds, doc='high value of beta')
    model.logbeta_high = Var(bounds=(0.0001, 5), doc='high value of beta')
    # dummy variables
    model.p = Var(model.S_meas, bounds=_gt_zero, doc='dummy variable p')
    model.n = Var(model.S_meas, bounds=_gt_zero, doc='dummy variable n')

    # define indexed constants

    # log of measured cases after adjusting for under-reporting
    logIstar = logIstar
    # changes in susceptible population profile from susceptible reconstruction
    deltaS = deltaS
    # mean susceptibles (Number of Population)
    # meanS = 1.04e6
    meanS = 8.65e5  # Number of Population(people)
    logN = pop  # log of measured population (Number of Population(log scale with log base 10))
    # define index for beta over all measurements ()
    beta_set = beta_set

    # define objective
    def _obj_rule(m):
        """
        Objective function for the SIR disease model, aiming to minimize the total discrepancy between estimated and observed infectious cases.

        Parameters
        ----------
        m : Pyomo.ConcreteModel
            SIR disease model using a low/high transmission parameter

        Returns
        -------
        Pyomo.Expression
            The expression for the objective function, which is the sum of overestimation and underestimation errors across all time periods.

        Notes
        -----
        These variables ('p' and 'n') are likely to represent the overestimation and underestimation errors, respectively,
        in the model's estimation of infectious cases compared to the observed data.
        By minimizing their sum, the model seeks to closely align its estimations with the actual observed data.
        """
        expr = sum(m.p[i] + m.n[i] for i in m.S_meas)
        return expr

    model.obj = Objective(rule=_obj_rule, sense=minimize, doc='objective function')

    # define constraints
    def _logSIR(m, i):
        """
        SIR model constraint capturing the dynamics of infectious disease spread using a logarithmic formulation.

        Parameters
        ----------
        m : Pyomo.ConcreteModel
            SIR disease model using a low/high transmission parameter
        i : int
            index of biweekly periods in the data set

        Returns
        -------
        tuple
            A tuple containing the constraint expression for the SIR dynamics at the i-th bi-weekly period.

        Notes
        -----
        This constraint is based on the differential equations of the SIR model, discretized and transformed into a logarithmic scale.
        The 0.0 in (0.0, expr) enforces expr to be zero, defining an equality constraint essential for the SIR model to accurately capture the exact dynamics of disease transmission between time steps.
        """
        expr = m.logI[i + 1] - (
            m.logbeta[beta_set[i - 1]]
            + m.logI[i]
            + math.log(deltaS[i - 1] + meanS)
            - logN[i - 1]
        )
        return (0.0, expr)

    model.logSIR = Constraint(
        model.S_meas_small, rule=_logSIR, doc='log of SIR disease model'
    )

    # objective function constraint
    def _p_n_const(m, i):
        """
        Defines a constraint relating the model's estimated infectious cases to observed data, adjusted for overestimation and underestimation.

        It includes the variables 'p' and 'n', which represent the overestimation and underestimation errors, respectively.

        Parameters
        ----------
        m : Pyomo.ConcreteModel
            SIR disease model using a low/high transmission parameter
        i : int
            index of biweekly periods in the data set

        Returns
        -------
        tuple
            A tuple containing the constraint expression for adjusting the model's estimated cases at the i-th bi-weekly period.

        Notes
        -----
        The constraint is formulated to account for the difference between the logarithm of observed infectious cases (logIstar) and the model's logarithm of estimated infectious cases (m.logI).
        The 'p' and 'n' variables in the model represent overestimation and underestimation errors, respectively, and this constraint helps in aligning the model's estimates with actual observed data.
        """
        expr = logIstar[i - 1] - m.logI[i] - m.p[i] + m.n[i]
        return (0.0, expr)

    model.p_n_const = Constraint(
        model.S_meas, rule=_p_n_const, doc='constraint for p and n'
    )

    # disjuncts

    model.BigM = Suffix()
    model.y = RangeSet(0, 1)

    def _high_low(disjunct, i, y):
        """
        Disjunct function for setting high and low beta values based on the binary variable.

        Parameters
        ----------
        disjunct : Pyomo.Disjunction
            The disjunct block being defined.
        i : int
            Index of biweekly periods in the data set.
        y : int
            Binary variable indicating whether the high or low beta value is used.

        Returns
        -------
        None
            Modifies the disjunct block to include the appropriate constraint based on the value of y.

        Notes
        -----
        This function contributes to the disjunctive formulation of the model, allowing for the selection between high and low transmission rates for the disease.
        """
        model = disjunct.model()
        if y:
            disjunct.c = Constraint(expr=model.logbeta_high - model.logbeta[i] == 0.0)
        else:
            disjunct.c = Constraint(expr=model.logbeta[i] - model.logbeta_low == 0.0)
        model.BigM[disjunct.c] = bigM

    model.high_low = Disjunct(
        model.S_beta,
        model.y,
        rule=_high_low,
        doc='disjunct for high and low beta values',
    )

    # disjunctions
    def _disj(model, i):
        """
        Defines a disjunction for each beta value to choose between high and low transmission rates.

        Parameters
        ----------
        model : Pyomo.Disjunction
            disjunction for the high and low beta values
        i : int
            Index of biweekly periods in the data set

        Returns
        -------
        list
            A list of disjuncts for the i-th biweekly period, enabling the model to choose between
            high and low beta values.
            Each disjunct represents a set of constraints that are activated
            based on the binary decision variables.

        Notes
        -----
        This list is used by Pyomo to create a disjunction for each biweekly period, allowing the model to choose between the high or low beta value constraints based on the optimization process.
        The defined disjunctions are integral to the model, enabling it to adaptively select the appropriate beta value for each time period, reflecting changes in disease transmission dynamics.
        """
        return [model.high_low[i, j] for j in model.y]

    model.disj = Disjunction(
        model.S_beta, rule=_disj, doc='disjunction for high and low beta values'
    )

    return model


# disjuncts
# The commented code sets up explicit high and low beta constraints for the SIR model using a big-M reformulation, bypassing Pyomo's built-in disjunctive programming tools.
# high beta disjuncts
# def highbeta_L(m,i):
#     """
#     Defines the lower bound constraint for the high transmission parameter beta in the SIR model.

#     Parameters
#     ----------
#     m : Pyomo.ConcreteModel
#         SIR disease model using a low/high transmission parameter
#     i : int
#         index of biweekly periods in the data set

#     Returns
#     -------
#     tuple
#         A tuple (0.0, expr, None) where expr is the Pyomo expression for the lower bound of the high beta disjunct at the i-th biweekly period.
#         This represents the lower bound of the high beta disjunct at the i-th biweekly period.

#     Notes
#     -----
#     The given function is given as the expression that the disjunctions are converted by big-M reformulation.
#     The binary variable m.y[i] is commented inside the model function.
#     """
#     expr = m.logbeta[i] - m.logbeta_high + bigM*(1-m.y[i])
#     return (0.0, expr, None)
# model.highbeta_L = Constraint(model.S_beta, rule=highbeta_L)

# def highbeta_U(m,i):
#     """
#     Defines the upper bound constraint for the high transmission parameter beta in the SIR model.

#     Parameters
#     ----------
#     m : Pyomo.ConcreteModel
#         SIR disease model using a low/high transmission parameter
#     i : int
#         Index of biweekly periods in the data set.

#     Returns
#     -------
#     tuple
#         A tuple (None, expr, 0.0) where expr is the Pyomo expression for the upper bound of the high beta disjunct at the i-th biweekly period.
#         This represents the upper bound of the high beta disjunct at the i-th biweekly period.

#     Notes
#         The given function is given as the expression that the disjunctions are converted by big-M reformulation.
#         The binary variable m.y[i] is commented inside the model function.
#     """
#     expr = m.logbeta[i] - m.logbeta_high
#     return (None, expr, 0.0)
# model.highbeta_U = Constraint(model.S_beta, rule=highbeta_U)

# # low beta disjuncts
# def lowbeta_U(m,i):
#     """
#     Defines the upper bound constraint for the low transmission parameter beta in the SIR model.

#     Parameters
#     ----------
#         m (Pyomo.ConcreteModel): SIR disease model using a low/high transmission parameter
#         i (int): index of biweekly periods in the data set

#     Returns:
#         A tuple (None, expr, 0.0) where expr is the Pyomo expression for the upper bound of the low beta disjunct at the i-th biweekly period.
#         This represents the upper bound of the low beta disjunct at the i-th biweekly period.

#     Notes
#     -----
#     The given function is given as the expression that the disjunctions are converted by big-M reformulation.
#     The binary variable m.y[i] is commented inside the model function.
#     """
#     expr = m.logbeta[i] - m.logbeta_low - bigM*(m.y[i])
#     return (None, expr, 0.0)
# model.lowbeta_U = Constraint(model.S_beta, rule=lowbeta_U)

# def lowbeta_L(m,i):
#     """
#     Defines the lower bound constraint for the low transmission parameter beta in the SIR model.

#     Parameters
#     ----------
#         m (Pyomo.ConcreteModel): SIR disease model using a low/high transmission parameter
#         i (int): index of biweekly periods in the data set

#     Returns
#     -------
#     tuple
#         A tuple (0.0, expr, None) where expr is the Pyomo expression for the lower bound of the low beta disjunct at the i-th biweekly period.
#         This represents the lower bound of the low beta disjunct at the i-th biweekly period.

#     Notes
#     -----
#     This lower bound constraint is part of the model's disjunctive framework, allowing for the differentiation between low and high transmission rates.
#     The big-M method integrates this constraint into the model based on the state of the binary decision variable `m.y[i]`.
#     """
#     expr = m.logbeta[i] - m.logbeta_low
#     return (0.0, expr, None)
# model.lowbeta_L = Constraint(model.S_beta, rule=lowbeta_L)


if __name__ == "__main__":
    m = build_model()
    TransformationFactory('gdp.bigm').apply_to(m)
    SolverFactory('gams').solve(
        m, solver='baron', tee=True, add_options=['option optcr=1e-6;']
    )
    m.obj.display()
