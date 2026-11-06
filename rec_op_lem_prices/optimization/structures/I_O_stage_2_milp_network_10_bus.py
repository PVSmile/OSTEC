"""
Input structures for the Stage 2 MILPs (bilateral and pool) with distribution network constraints
(Relaxed Branch Flow Model).

Network: TC6 radial distribution network, 10 buses, 9 branches (23 kV, 10 MVA base).
Data source: agents_data.xlsx (TC6 case study).
Community members: buses 2..10 (bus 1 = slack/grid connection point).
Horizon: 24 hours, delta_t = 1 h.
"""

import math

# ---------------------------------------------------------------------------
# NETWORK TOPOLOGY AND PARAMETERS (TC6, 23 kV, 10 MVA base)
# ---------------------------------------------------------------------------

_NODES  = [str(i) for i in range(1, 11)]
_LINES  = [("1","2"),("2","3"),("3","4"),("4","5"),("5","6"),
           ("6","7"),("7","8"),("8","9"),("9","10")]
_SLACK  = "1"
_MEMBERS = [n for n in _NODES if n != _SLACK]  # buses 2..10

_S_base_MVA = 10.0
_V_base_kV  = 23.0
_Z_base     = (_V_base_kV ** 2) / _S_base_MVA          # 52.9 Ω
_S_base_kW  = _S_base_MVA * 1000.0                      # 10 000 kW
_I_base_A   = (_S_base_MVA * 1e6) / (math.sqrt(3) * _V_base_kV * 1e3)  # ≈ 251.02 A
_Imax_A     = 150.0
_Imax_pu    = _Imax_A / _I_base_A                        # ≈ 0.5976 pu

# Branch resistances and reactances [Ω] (from TC6 PowerWorld data)
_R_ohm = {
    ("1","2"):  0.00233081285444234,
    ("2","3"):  0.000264650283553875,
    ("3","4"):  0.0141077504725898,
    ("4","5"):  0.0132022684310019,
    ("5","6"):  0.0374877126654064,
    ("6","7"):  0.0171134215500945,
    ("7","8"):  0.0388506616257089,
    ("8","9"):  0.090648393194707,
    ("9","10"): 0.101009451795841,
}
_X_ohm = {
    ("1","2"):  0.00780151228733459,
    ("2","3"):  0.0114385633270321,
    ("3","4"):  0.0227788279773157,
    ("4","5"):  0.0115009451795841,
    ("5","6"):  0.0326578449905482,
    ("6","7"):  0.0149073724007561,
    ("7","8"):  0.0220037807183365,
    ("8","9"):  0.0513421550094518,
    ("9","10"): 0.0572098298676749,
}

_R = {ij: r / _Z_base for ij, r in _R_ohm.items()}
_X = {ij: x / _Z_base for ij, x in _X_ohm.items()}

_Vmin = {n: (1.0 if n == _SLACK else 0.95) for n in _NODES}
_Vmax = {n: (1.0 if n == _SLACK else 1.05) for n in _NODES}
_Imax = {ij: _Imax_pu for ij in _LINES}

_grid_data = {
    "nodes":          _NODES,
    "lines":          _LINES,
    "R":              _R,
    "X":              _X,
    "Vmin":           _Vmin,
    "Vmax":           _Vmax,
    "Imax":           _Imax,
    "slack":          _SLACK,
    "S_base":         _S_base_MVA,
    "S_base_kW":      _S_base_kW,
    "V_base_kV":      _V_base_kV,
    "Pslack_max_pu":  10.0,
    "Qslack_max_pu":  10.0,
}

# ---------------------------------------------------------------------------
# TARIFF VECTORS (24 h, same for all buses — from pi_sup / pi_sur / pi_lem sheets)
# ---------------------------------------------------------------------------

_l_buy = [
    0.145,   0.14245, 0.138,   0.13505, 0.1354,  0.14,
    0.13999, 0.145,   0.1479,  0.14306, 0.14245, 0.14245,
    0.14194, 0.14294, 0.14306, 0.14306, 0.14315, 0.14315,
    0.14315, 0.14999, 0.175,   0.18401, 0.18006, 0.16432,
]

_l_sell = [
    0.058,    0.05698,  0.0552,   0.05402,  0.05416,  0.056,
    0.055996, 0.058,    0.05916,  0.057224, 0.05698,  0.05698,
    0.056776, 0.057176, 0.057224, 0.057224, 0.05726,  0.05726,
    0.05726,  0.059996, 0.07,     0.073604, 0.072024, 0.065728,
]

_l_lem = [
    0.1305,   0.128205, 0.1242,   0.121545, 0.12186,  0.126,
    0.125991, 0.1305,   0.13311,  0.128754, 0.128205, 0.128205,
    0.127746, 0.128646, 0.128754, 0.128754, 0.128835, 0.128835,
    0.128835, 0.134991, 0.1575,   0.165609, 0.162054, 0.147888,
]

# ---------------------------------------------------------------------------
# ACTIVE AND REACTIVE LOAD PROFILES PER BUS (from Pc, Qc, PV sheets)
# Values are in kWh / kVArh per period (delta_t = 1 h → kWh = kW, kVArh = kVAr)
# ---------------------------------------------------------------------------

# Active load [kW] → e_c [kWh] (×delta_t = 1 h)
_Pc = {
    "2":  [87.0, 6.0, 6.0, 3.0, 21.0, 6.0, 3.0, 3.0, 6.0, 24.0, 186.0, 123.0,
           114.0, 96.0, 93.0, 96.0, 99.0, 87.0, 90.0, 168.0, 165.0, 114.0, 96.0, 39.0],
    "3":  [87.0, 6.0, 6.0, 3.0, 21.0, 6.0, 3.0, 3.0, 6.0, 24.0, 186.0, 123.0,
           114.0, 96.0, 93.0, 96.0, 99.0, 87.0, 90.0, 168.0, 165.0, 114.0, 96.0, 39.0],
    "4":  [55.0, 33.0, 33.0, 17.6, 17.6, 17.6, 17.6, 17.6, 55.0, 88.0, 77.0, 101.2,
           138.6, 178.2, 154.0, 169.4, 165.0, 187.0, 156.2, 160.6, 162.8, 123.2, 94.6, 50.6],
    "5":  [55.0, 33.0, 33.0, 17.6, 17.6, 17.6, 17.6, 17.6, 55.0, 88.0, 77.0, 101.2,
           138.6, 178.2, 154.0, 169.4, 165.0, 187.0, 156.2, 160.6, 162.8, 123.2, 94.6, 50.6],
    "6":  [22.0, 22.0, 20.0, 16.0, 18.0, 16.0, 32.0, 26.0, 22.0, 42.0, 26.0, 40.0,
           28.0, 50.0, 42.0, 48.0, 68.0, 50.0, 68.0, 48.0, 28.0, 20.0, 22.0, 20.0],
    "7":  [22.0, 22.0, 20.0, 16.0, 18.0, 16.0, 32.0, 26.0, 22.0, 42.0, 26.0, 40.0,
           28.0, 50.0, 42.0, 48.0, 68.0, 50.0, 68.0, 48.0, 28.0, 20.0, 22.0, 20.0],
    "8":  [16.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 40.0, 80.0, 88.0, 184.0, 224.0,
           196.0, 188.0, 168.0, 108.0, 104.0, 108.0, 144.0, 124.0, 96.0, 108.0, 104.0, 76.0],
    "9":  [38.4, 38.4, 16.0, 16.0, 16.0, 16.0, 19.2, 19.2, 16.0, 12.8, 41.6, 48.0,
           9.6, 9.6, 70.4, 108.8, 96.0, 166.4, 166.4, 137.6, 156.8, 169.6, 169.6, 172.8],
    "10": [38.4, 38.4, 16.0, 16.0, 16.0, 16.0, 19.2, 19.2, 16.0, 12.8, 41.6, 48.0,
           9.6, 9.6, 70.4, 108.8, 96.0, 166.4, 166.4, 137.6, 156.8, 169.6, 169.6, 172.8],
}

# Reactive load [kVAr] → q_c [kVArh] (Qc = 20% of Pc, from Qc sheet)
_Qc = {
    "2":  [17.4, 1.2, 1.2, 0.6, 4.2, 1.2, 0.6, 0.6, 1.2, 4.8, 37.2, 24.6,
           22.8, 19.2, 18.6, 19.2, 19.8, 17.4, 18.0, 33.6, 33.0, 22.8, 19.2, 7.8],
    "3":  [17.4, 1.2, 1.2, 0.6, 4.2, 1.2, 0.6, 0.6, 1.2, 4.8, 37.2, 24.6,
           22.8, 19.2, 18.6, 19.2, 19.8, 17.4, 18.0, 33.6, 33.0, 22.8, 19.2, 7.8],
    "4":  [11.0, 6.6, 6.6, 3.52, 3.52, 3.52, 3.52, 3.52, 11.0, 17.6, 15.4, 20.24,
           27.72, 35.64, 30.8, 33.88, 33.0, 37.4, 31.24, 32.12, 32.56, 24.64, 18.92, 10.12],
    "5":  [11.0, 6.6, 6.6, 3.52, 3.52, 3.52, 3.52, 3.52, 11.0, 17.6, 15.4, 20.24,
           27.72, 35.64, 30.8, 33.88, 33.0, 37.4, 31.24, 32.12, 32.56, 24.64, 18.92, 10.12],
    "6":  [4.4, 4.4, 4.0, 3.2, 3.6, 3.2, 6.4, 5.2, 4.4, 8.4, 5.2, 8.0,
           5.6, 10.0, 8.4, 9.6, 13.6, 10.0, 13.6, 9.6, 5.6, 4.0, 4.4, 4.0],
    "7":  [4.4, 4.4, 4.0, 3.2, 3.6, 3.2, 6.4, 5.2, 4.4, 8.4, 5.2, 8.0,
           5.6, 10.0, 8.4, 9.6, 13.6, 10.0, 13.6, 9.6, 5.6, 4.0, 4.4, 4.0],
    "8":  [3.2, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 8.0, 16.0, 17.6, 36.8, 44.8,
           39.2, 37.6, 33.6, 21.6, 20.8, 21.6, 28.8, 24.8, 19.2, 21.6, 20.8, 15.2],
    "9":  [7.68, 7.68, 3.2, 3.2, 3.2, 3.2, 3.84, 3.84, 3.2, 2.56, 8.32, 9.6,
           1.92, 1.92, 14.08, 21.76, 19.2, 33.28, 33.28, 27.52, 31.36, 33.92, 33.92, 34.56],
    "10": [7.68, 7.68, 3.2, 3.2, 3.2, 3.2, 3.84, 3.84, 3.2, 2.56, 8.32, 9.6,
           1.92, 1.92, 14.08, 21.76, 19.2, 33.28, 33.28, 27.52, 31.36, 33.92, 33.92, 34.56],
}

# PV generation [kW] → e_g [kWh] (buses 2, 5, 8, 10 have PV; others = 0)
_PV_profile = [
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    1.292, 7.024, 17.304, 22.129, 52.646, 72.518,
    95.572, 153.756, 137.701, 60.801, 79.296, 72.969,
    45.469, 16.474, 3.227, 0.0, 0.0, 0.0,
]
_PV = {
    "2":  _PV_profile,
    "3":  [0.0] * 24,
    "4":  [0.0] * 24,
    "5":  _PV_profile,
    "6":  [0.0] * 24,
    "7":  [0.0] * 24,
    "8":  _PV_profile,
    "9":  [0.0] * 24,
    "10": _PV_profile,
}

# max_p per meter: 1.2 × max(|Pc|, |PV|) [kW]
_max_p = {
    n: 1.2 * max(max(abs(p) for p in _Pc[n]), max(abs(g) for g in _PV[n]))
    for n in _MEMBERS
}

# l_grid: access tariff between pairs of community members [€/kWh] — 0 for P2P
_l_grid = {
    n: {m: [0.0] * 24 for m in _MEMBERS if m != n}
    for n in _MEMBERS
}

# ---------------------------------------------------------------------------
# INPUTS DICTIONARY (passed as backpack to StageTwoMILPBilateral)
# ---------------------------------------------------------------------------

INPUTS_S2_BILATERAL_NETWORK_10_BUS = {
    # Temporal
    'delta_t': 1.0,          # settlement interval [h]
    'horizon': 24.0,         # planning horizon [h]
    # Market/tariff parameters
    'l_extra':       10,     # penalty for violating p_meter_max [€/kW]
    'l_dr':          5.0,    # demand-response curtailment penalty [€/kWh]
    'l_gc':          0.11,   # generation curtailment penalty [€/kWh]
    'l_penalty':     1000.0, # penalty for no-worse-off slack [€/€]
    'l_grid':        _l_grid,
    'l_lem':         _l_lem,
    'l_market_buy':  _l_lem,  # market-indexed buying price = LEM price
    'l_market_sell': _l_lem,  # market-indexed selling price = LEM price
    # Meter data (one entry per community member / bus)
    'meters': {
        n: {
            'btm_storage': None,      # no battery storage in TC6 base case
            'btm_evs':     None,      # no EVs in TC6 base case
            'ewh':         None,
            'hvac':        None,
            'hp':          None,
            'c_ind':       0.0,       # placeholder; substituído pelo Stage 1 no teste
            'e_c':         _Pc[n],    # active load [kWh/period]
            'e_g':         _PV[n],    # PV generation [kWh/period]
            'q_c':         _Qc[n],    # reactive load [kVArh/period]
            'max_p':       _max_p[n], # meter power limit [kW]
            'l_buy':       _l_buy,    # retail buy tariff [€/kWh]
            'l_sell':      _l_sell,   # feed-in tariff [€/kWh]
        }
        for n in _MEMBERS
    },
    # Stage flags
    'second_stage':      True,
    'strict_pos_coeffs': True,
    'total_share_coeffs': False,
    # Network (grid) data
    'grid_data': _grid_data,
}

# ---------------------------------------------------------------------------
# INPUTS DICTIONARY FOR THE POOL MARKET (same community and network)
# ---------------------------------------------------------------------------

INPUTS_S2_POOL_NETWORK_10_BUS = {
    k: v for k, v in INPUTS_S2_BILATERAL_NETWORK_10_BUS.items() if k != 'l_grid'
}
INPUTS_S2_POOL_NETWORK_10_BUS['l_grid'] = [0.0] * 24  # access tariff for local self-consumption [€/kWh]
