"""
Tests for the Stage 2 MILPs (bilateral and pool) with distribution network constraints
(Relaxed Branch Flow Model with SOC cuts), on the 10-bus TC6 network and using the default solver (CBC).
"""
import copy
import functools
import math

import pytest

from rec_op_lem_prices.optimization_functions import (
	run_pre_two_stage_collective_bilateral_milp,
	run_pre_two_stage_collective_pool_milp
)
from rec_op_lem_prices.optimization.module.StageOneMILP import StageOneMILP
from rec_op_lem_prices.optimization.module.StageTwoMILPBilateral import StageTwoMILPBilateral
from rec_op_lem_prices.optimization.module.StageTwoMILPPool import StageTwoMILPPool
from rec_op_lem_prices.optimization.structures.I_O_stage_2_milp_network_10_bus import (
	INPUTS_S2_BILATERAL_NETWORK_10_BUS,
	INPUTS_S2_POOL_NETWORK_10_BUS
)

MARKETS = {
	'bilateral': (StageTwoMILPBilateral, INPUTS_S2_BILATERAL_NETWORK_10_BUS, 'c_ind2bilateral'),
	'pool': (StageTwoMILPPool, INPUTS_S2_POOL_NETWORK_10_BUS, 'c_ind2pool'),
}
TOL = 1e-5


def _stress(inputs: dict, load_factor: float) -> dict:
	"""Scales the active and reactive loads, updating the meters' power limits accordingly."""
	for md in inputs['meters'].values():
		md['e_c'] = [v * load_factor for v in md['e_c']]
		md['q_c'] = [v * load_factor for v in md['q_c']]
		md['max_p'] = 1.2 * max(max(abs(v) for v in md['e_c']), max(abs(v) for v in md['e_g']))
	return inputs


@functools.lru_cache(maxsize=None)
def _stage1_costs(load_factor: float) -> dict:
	"""Stage 1 (individual) cost of each member; the network does not take part in Stage 1."""
	inputs = _stress(copy.deepcopy(INPUTS_S2_BILATERAL_NETWORK_10_BUS), load_factor)
	c_ind = {}
	for n, md in inputs['meters'].items():
		backpack = {
			'id': n, 'delta_t': inputs['delta_t'], 'horizon': inputs['horizon'],
			'e_c': md['e_c'], 'e_g': md['e_g'], 'l_buy': md['l_buy'], 'l_sell': md['l_sell'],
			'l_market_buy': inputs['l_market_buy'], 'l_market_sell': inputs['l_market_sell'],
			'l_extra': inputs['l_extra'], 'max_p': md['max_p'], 'btm_storage': None, 'btm_evs': None,
			'ewh': None, 'hvac': None, 'hp': None,
		}
		milp = StageOneMILP(backpack)
		milp.solve_milp()
		assert milp.status == 'Optimal', f'Stage 1 not optimal for member {n}'
		c_ind[n] = milp.obj_value
	return c_ind


@functools.lru_cache(maxsize=None)
def _solve(market: str, load_factor: float):
	cls, base_inputs, _ = MARKETS[market]
	inputs = _stress(copy.deepcopy(base_inputs), load_factor)
	for n, c in _stage1_costs(load_factor).items():
		inputs['meters'][n]['c_ind'] = c
	milp = cls(inputs)
	# accept_tol=0: iterate the SOC cuts until no violation above tol remains
	milp.solve_milp_soc_cuts(accept_tol=0)
	assert milp.status == 'Optimal', f'{market}, load x{load_factor}: {milp.status}'
	return inputs, milp.generate_outputs()


def _assert_network_limits(inputs: dict, results: dict):
	grid = inputs['grid_data']
	n_steps = len(results['P_slack'])
	# Voltages within limits and slack voltage at 1 pu
	for n, v in results['v_node'].items():
		for t in range(n_steps):
			assert grid['Vmin'][n] - TOL <= v[t] <= grid['Vmax'][n] + TOL, f'voltage at node {n}, t={t}'
	assert all(v == pytest.approx(1.0, abs=TOL) for v in results['v_node'][grid['slack']])
	# Currents within limits and SOC relaxation tight (P² + Q² ≈ v·l)
	for (n, m), i_pct in results['I_branch_pct'].items():
		for t in range(n_steps):
			assert i_pct[t] <= 100 + 1e-3, f'current in branch {(n, m)}, t={t}'
			p, q = results['P_branch'][(n, m)][t], results['Q_branch'][(n, m)][t]
			v2, i2 = results['v_node'][n][t] ** 2, results['I_branch'][(n, m)][t] ** 2
			assert p * p + q * q - v2 * i2 <= TOL, f'SOC violation in branch {(n, m)}, t={t}'
	# Slack injection equals the community's net consumption plus the (small) network losses
	s_base = grid['S_base_kW']
	for t in range(n_steps):
		net_load = sum(results['e_cmet'][n][t] for n in inputs['meters']) / (inputs['delta_t'] * s_base)
		losses = sum(grid['R'][(n, m)] * results['I_branch'][(n, m)][t] ** 2 for (n, m) in grid['lines'])
		assert results['P_slack'][t] == pytest.approx(net_load + losses, abs=1e-4), f't={t}'


@pytest.mark.parametrize('market', ['bilateral', 'pool'])
def test_network_base_case(market):
	inputs, results = _solve(market, 1.0)
	_assert_network_limits(inputs, results)
	# Uncongested network: no curtailment and no member pays more than in Stage 1
	assert sum(sum(v) for v in results['e_c_cut'].values()) == pytest.approx(0.0, abs=TOL)
	assert sum(sum(v) for v in results['e_g_cut'].values()) == pytest.approx(0.0, abs=TOL)
	assert all(d == pytest.approx(0.0, abs=TOL) for d in results['deltaY'].values())
	# Each member's energy cost does not exceed its Stage 1 cost; the objective adds the (positive) cost of losses
	cost_key = MARKETS[market][2]
	c_ind = _stage1_costs(1.0)
	assert all(results[cost_key][n] <= c_ind[n] + 1e-3 for n in c_ind)
	assert sum(results[cost_key].values()) == pytest.approx(1645.5696, abs=1e-2)
	assert results['obj_value'] > sum(results[cost_key].values())


@pytest.mark.parametrize('market', ['bilateral', 'pool'])
def test_network_congested_case(market):
	inputs, results = _solve(market, 6.0)
	_assert_network_limits(inputs, results)
	# With 6x the load, the current limits become binding and load has to be curtailed
	assert max(max(v) for v in results['I_branch_pct'].values()) == pytest.approx(100.0, abs=1e-2)
	assert sum(sum(v) for v in results['e_c_cut'].values()) > 1.0


@pytest.mark.parametrize('load_factor', [1.0, 6.0])
def test_network_pool_equals_bilateral(load_factor):
	# The network sees only the members' net consumption, and with l_grid = 0 both markets reach the same optimum
	_, res_bilateral = _solve('bilateral', load_factor)
	_, res_pool = _solve('pool', load_factor)
	assert res_pool['obj_value'] == pytest.approx(res_bilateral['obj_value'], rel=1e-6)
	for key in ('e_c_cut', 'e_g_cut'):
		assert sum(sum(v) for v in res_pool[key].values()) == pytest.approx(
			sum(sum(v) for v in res_bilateral[key].values()), abs=1e-3)
	for t, p in enumerate(res_pool['P_slack']):
		assert p == pytest.approx(res_bilateral['P_slack'][t], abs=1e-4)


@pytest.mark.parametrize('market', ['bilateral', 'pool'])
def test_without_grid_no_network_terms(market):
	# Without grid_data, solve_milp_soc_cuts solves the original MILP (no network, curtailment or deltaY);
	# the small (3-step) cases of the existing Stage 2 tests are used, since the 24-step bilateral case
	# without the network bounds is too heavy for CBC
	from rec_op_lem_prices.optimization.structures.I_O_stage_2_bilateral_milp import INPUTS_S2_BILATERAL
	from rec_op_lem_prices.optimization.structures.I_O_stage_2_pool_milp import INPUTS_S2_POOL
	cls, _, cost_key = MARKETS[market]
	base_inputs = INPUTS_S2_BILATERAL if market == 'bilateral' else INPUTS_S2_POOL

	milp_soc = cls(copy.deepcopy(base_inputs))
	milp_soc.solve_milp_soc_cuts()
	assert milp_soc.status == 'Optimal'
	results = milp_soc.generate_outputs()
	for key in ('e_c_cut', 'e_g_cut', 'q_g', 'deltaY', 'P_branch', 'v_node', 'P_slack'):
		assert key not in results

	milp = cls(copy.deepcopy(base_inputs))
	milp.solve_milp()
	assert milp.status == 'Optimal'
	assert results['obj_value'] == pytest.approx(milp.obj_value, abs=1e-6)
	assert results[cost_key] == pytest.approx(milp.generate_outputs()[cost_key], abs=1e-6)


@pytest.mark.parametrize('market, run_function', [
	('bilateral', run_pre_two_stage_collective_bilateral_milp),
	('pool', run_pre_two_stage_collective_pool_milp),
])
def test_network_through_optimization_functions(market, run_function):
	# grid_data passed to the public two-stage functions: solve_milp applies the SOC cuts, so that the
	# network limits are respected and the results match the ones of solve_milp_soc_cuts
	_, base_inputs, cost_key = MARKETS[market]
	inputs = _stress(copy.deepcopy(base_inputs), 6.0)
	results, stage1_results = run_function(inputs, for_testing=True)
	assert results['milp_status'] == 'Optimal'
	assert all(r['milp_status'] == 'Optimal' for r in stage1_results)
	_assert_network_limits(inputs, results)
	assert max(max(v) for v in results['I_branch_pct'].values()) == pytest.approx(100.0, abs=1e-2)

	_, reference = _solve(market, 6.0)
	assert results['obj_value'] == pytest.approx(reference['obj_value'], rel=1e-6)
