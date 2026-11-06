"""
Helpers for the optional distribution network constraints of the Stage 2 MILPs (pool and bilateral).

The network is modelled with the Relaxed Branch Flow Model (DistFlow) of a radial network:
	- v[n][t]      squared voltage magnitude at node n [pu²]
	- l[(n,m)][t]  squared current magnitude in branch (n, m) [pu²]
	- P/Q[(n,m)][t] active/reactive power flowing from n to m [pu]
The non-convex relation P² + Q² = v·l is relaxed to the second-order cone P² + Q² ≤ v·l, which is
enforced through an iterative outer approximation (linear cutting planes), so that the problem
remains a MILP that can be solved with CBC.

All network variables are named with the prefix NETWORK_VAR_PREFIX, so that they can be easily
skipped when the MILP outputs are generated.
"""
import math

from loguru import logger
from pulp import (
	LpStatus,
	LpVariable,
	lpSum,
	value
)

NETWORK_VAR_PREFIX = 'net_'
DEFAULT_LOSS_COST = 1.0  # cost of the network losses [€/kWh]; can be changed with grid_data['loss_cost']
RELAXATION_GAP_TOL = 1e-4  # maximum v·l - (P² + Q²) [pu²] before warning that the SOC relaxation is not exact


def _symmetric(param: dict, lines: list) -> dict:
	"""
	Returns a copy of a branch parameter dictionary that is defined for both directions of every line.
	"""
	sym = dict(param)
	for (n, m) in lines:
		if (m, n) not in sym:
			sym[(m, n)] = sym[(n, m)]
	return sym


def local_trade_bounds(milp_model, n: str, t: int, ewh_original_load=None) -> tuple:
	"""
	Maximum energy that meter n can buy from / sell to the local market in time step t [kWh].
	The purchases cannot exceed the maximum possible deficit of the meter (all load, with all generation
	curtailed, plus the maximum consumption of its flexible assets) and the sales cannot exceed its maximum
	possible surplus (all generation, with all load curtailed, plus the maximum injection of its flexible
	assets). These bounds exclude "virtual" transactions (buying locally only to resell) and greatly
	tighten the MILP, which is essential for the solution time when the network is considered.
	:param milp_model: Stage 2 MILP object (pool or bilateral) with the assets' data already unpacked
	:param n: meter ID
	:param t: time step
	:param ewh_original_load: optional dict {ewh_id: original load profile} subtracted from e_cmet
	:return: tuple with (maximum purchase, maximum sale) [kWh]
	"""
	m = milp_model
	dt = m._delta_t
	storage = sum(m._p_max[n][b] * dt for b in m.sets_btm_storage[n])
	ev_charge = sum(m._pmax_c_ev[n][ev] / m._eff_bc_ev[n][ev] * m._bin_ev[n][ev][t] * dt for ev in m.sets_btm_ev[n])
	ev_discharge = sum(m._pmax_d_ev[n][ev] * m._eff_bd_ev[n][ev] * m._bin_ev[n][ev][t] * dt for ev in m.sets_btm_ev[n])
	hvac = sum(m.hvac_capacity[n][h] * dt for h in m.set_hvac.get(n, []))
	hp = sum(m.hp_power_rated[n][hp] * max(1.0, dt) for hp in m.set_hp.get(n, []))
	ewh = sum(m.ewh_power[n][e] * m.delta_t[n][e] for e in m.set_ewh.get(n, []))
	ewh_orig = sum(load[t] for load in (ewh_original_load or {}).values())

	pur_max = max(0.0, m._e_c[n][t]) + storage + ev_charge + hvac + hp + ewh
	sale_max = max(0.0, m._e_g[n][t]) + storage + ev_discharge + ewh_orig
	return pur_max, sale_max


def add_network_constraints(milp, grid_data: dict, time_series, delta_t: float, set_meters: list,
							e_cmet: dict, q_g: dict, q_load: dict) -> dict:
	"""
	Adds the Relaxed Branch Flow Model variables and constraints to the MILP.
	The net active injection of each community member is given by its net consumption e_cmet (a positive
	e_cmet is a load); the reactive injection is the reactive generation q_g minus the reactive load q_load.
	Nodes that are not community members have no injection; the slack node exchanges power with the
	upstream grid.
	:param milp: puLP problem to which the network is added
	:param grid_data: network topology and parameters
	:param time_series: range with the time steps of the MILP
	:param delta_t: duration of each time step [h]
	:param set_meters: community members; each one must be a node of the network
	:param e_cmet: net consumption of each member per time step [kWh] (puLP variables)
	:param q_g: reactive generation of each member per time step [kVArh] (puLP variables)
	:param q_load: reactive load of each member per time step [kVArh] (numbers or puLP expressions)
	:return: dictionary with the network variables, used for the SOC cuts and outputs
	"""
	nodes = grid_data['nodes']
	lines = grid_data['lines']
	slack = grid_data['slack']
	v_min = grid_data['Vmin']
	v_max = grid_data['Vmax']
	r = _symmetric(grid_data['R'], lines)
	x = _symmetric(grid_data['X'], lines)
	i_max = _symmetric(grid_data['Imax'], lines)
	s_base_kw = grid_data.get('S_base_kW', 10000.0)

	missing = [n for n in set_meters if n not in nodes]
	if missing:
		raise ValueError(f'The following community members are not nodes of the network: {missing}')

	lines_out_of = {n: [] for n in nodes}
	lines_into = {n: [] for n in nodes}
	for (n, m) in lines:
		lines_out_of[n].append((n, m))
		lines_into[m].append((n, m))

	p = NETWORK_VAR_PREFIX
	# Squared voltage magnitude [pu²], bounded by [Vmin², Vmax²]
	v = {n: {t: LpVariable(f'{p}v_{n}_t{t:03d}', lowBound=v_min[n] ** 2, upBound=v_max[n] ** 2)
			 for t in time_series} for n in nodes}
	# Squared current magnitude [pu²], bounded by [0, Imax²]
	l = {(n, m): {t: LpVariable(f'{p}l_{n}_{m}_t{t:03d}', lowBound=0, upBound=i_max[(n, m)] ** 2)
				  for t in time_series} for (n, m) in lines}
	# Branch active and reactive power flows [pu]
	P = {(n, m): {t: LpVariable(f'{p}P_{n}_{m}_t{t:03d}') for t in time_series} for (n, m) in lines}
	Q = {(n, m): {t: LpVariable(f'{p}Q_{n}_{m}_t{t:03d}') for t in time_series} for (n, m) in lines}
	# Power exchanged with the upstream grid at the slack node [pu]
	p_slack_max = grid_data.get('Pslack_max_pu', 10.0)
	# By default the community can export to the upstream grid as much as it can import (consistent with the
	# surplus sold to the retailer/market); set Pslack_min_pu = 0 to forbid any export
	p_slack_min = grid_data.get('Pslack_min_pu', -p_slack_max)
	q_slack_max = grid_data.get('Qslack_max_pu', 10.0)
	p_slack = {t: LpVariable(f'{p}Pslack_t{t:03d}', lowBound=p_slack_min, upBound=p_slack_max) for t in time_series}
	q_slack = {t: LpVariable(f'{p}Qslack_t{t:03d}', lowBound=-q_slack_max, upBound=q_slack_max) for t in time_series}

	# Cost of the network losses [€/kWh]. The SOC relaxation is only exact if increasing l has a cost: otherwise,
	# when it is cheaper than curtailing (e.g., to avoid over-voltages or a forbidden export), the MILP "burns"
	# power in fictitious losses (l > (P² + Q²) / v) and the voltages/currents found are not physical.
	loss_cost = grid_data.get('loss_cost', DEFAULT_LOSS_COST)
	milp.objective += loss_cost * s_base_kw * delta_t * lpSum(
		r[(n, m)] * l[(n, m)][t] for (n, m) in lines for t in time_series)
	# Small penalty on the squared currents to avoid degenerate solutions
	milp.objective += 1e-3 * lpSum(l[(n, m)][t] for (n, m) in lines for t in time_series)

	for t in time_series:
		# Slack node voltage fixed at 1.0 pu
		milp += v[slack][t] == 1.0, f'Net_slack_voltage_t{t:03d}'

		# Nodal active and reactive power balance (DistFlow)
		for n in nodes:
			if n == slack:
				p_inj = p_slack[t]
				q_inj = q_slack[t]
			elif n in set_meters:
				p_inj = -e_cmet[n][t] / (delta_t * s_base_kw)
				q_inj = (q_g[n][t] - q_load[n][t]) / (delta_t * s_base_kw)
			else:
				p_inj = 0.0
				q_inj = 0.0

			milp += (
				p_inj == lpSum(P[(n, m)][t] for (n, m) in lines_out_of[n])
				- lpSum(P[(m, n)][t] - r[(m, n)] * l[(m, n)][t] for (m, n) in lines_into[n])
			), f'Net_active_balance_{n}_t{t:03d}'

			milp += (
				q_inj == lpSum(Q[(n, m)][t] for (n, m) in lines_out_of[n])
				- lpSum(Q[(m, n)][t] - x[(m, n)] * l[(m, n)][t] for (m, n) in lines_into[n])
			), f'Net_reactive_balance_{n}_t{t:03d}'

		for (n, m) in lines:
			# Voltage drop along the branch
			milp += (
				v[m][t] == v[n][t] - 2 * (r[(n, m)] * P[(n, m)][t] + x[(n, m)] * Q[(n, m)][t])
				+ (r[(n, m)] ** 2 + x[(n, m)] ** 2) * l[(n, m)][t]
			), f'Net_voltage_drop_{n}_{m}_t{t:03d}'

			# Apparent power bounds (keep P and Q bounded before the SOC cuts are added)
			s_max = v_max[n] * i_max[(n, m)]
			milp += P[(n, m)][t] <= s_max, f'Net_P_max_{n}_{m}_t{t:03d}'
			milp += P[(n, m)][t] >= -s_max, f'Net_P_min_{n}_{m}_t{t:03d}'
			milp += Q[(n, m)][t] <= s_max, f'Net_Q_max_{n}_{m}_t{t:03d}'
			milp += Q[(n, m)][t] >= -s_max, f'Net_Q_min_{n}_{m}_t{t:03d}'

	return {'nodes': nodes, 'lines': lines, 'P': P, 'Q': Q, 'v': v, 'l': l,
			'P_slack': p_slack, 'Q_slack': q_slack, 'Imax': i_max}


def add_soc_cuts(milp, grid_vars: dict, time_series, tol=1e-6, max_cuts_per_iter=200, tag='soc') -> tuple:
	"""
	Adds linear outer-approximation cuts (supporting hyperplanes of the rotated second-order cone)
	at the points where the last solution violates P² + Q² ≤ v·l.
	:return: tuple with the number of cuts added and the maximum violation found
	"""
	P, Q, v, l = grid_vars['P'], grid_vars['Q'], grid_vars['v'], grid_vars['l']

	violations = []
	for (n, m) in grid_vars['lines']:
		for t in time_series:
			p0, q0, v0, l0 = P[(n, m)][t].value(), Q[(n, m)][t].value(), v[n][t].value(), l[(n, m)][t].value()
			if p0 is None or q0 is None or v0 is None or l0 is None:
				continue
			viol = p0 * p0 + q0 * q0 - v0 * l0
			if viol > tol:
				violations.append((viol, n, m, t, p0, q0, v0, l0))

	if not violations:
		return 0, 0.0

	violations.sort(reverse=True, key=lambda x: x[0])
	if max_cuts_per_iter is not None:
		violations = violations[:max_cuts_per_iter]

	n_added = 0
	for (viol, n, m, t, p0, q0, v0, l0) in violations:
		# Supporting hyperplane of ||[2P, 2Q, v - l]|| ≤ v + l at the point u0 = [2P0, 2Q0, v0 - l0]
		u1, u2, u3 = 2.0 * p0, 2.0 * q0, v0 - l0
		norm_u = math.sqrt(u1 * u1 + u2 * u2 + u3 * u3)
		if norm_u <= 1e-12:
			continue
		lhs = u1 * (2.0 * P[(n, m)][t]) + u2 * (2.0 * Q[(n, m)][t]) + u3 * (v[n][t] - l[(n, m)][t])
		rhs = norm_u * (v[n][t] + l[(n, m)][t])
		milp += lhs <= rhs, f'Net_{tag}_cut_{n}_{m}_t{t:03d}_{n_added:04d}'
		n_added += 1

	return n_added, violations[0][0]


def solve_with_soc_cuts(milp, solver, grid_vars, time_series, max_iters=30, tol=5e-7, max_cuts_per_iter=300,
						accept_tol=1e-3) -> tuple:
	"""
	Solves the MILP and, if a network is present, iteratively adds SOC cuts and re-solves until no cut is
	needed (maximum violation ≤ tol), the violation is negligible (≤ accept_tol) or max_iters is reached.
	:return: tuple with the solver status and the objective function value
	"""
	status = None
	for k in range(max_iters):
		try:
			milp.solve(solver)
		except Exception as e:
			logger.warning(f'Solver raised an error: \'{e}\'. Considering problem as "Infeasible".')
			return 'Infeasible', None

		status = LpStatus[milp.status]
		if status not in ('Optimal', 'Feasible') or grid_vars is None:
			break

		n_added, max_viol = add_soc_cuts(milp, grid_vars, time_series, tol=tol,
										 max_cuts_per_iter=max_cuts_per_iter, tag=f'soc{k}')
		logger.debug(f'-- SOC cuts: iteration={k}, cuts added={n_added}, max violation={max_viol:.3e}')
		if n_added == 0 or max_viol <= accept_tol:
			break

	try:
		opt_value = value(milp.objective)
	except Exception:
		opt_value = None

	if grid_vars is not None and status in ('Optimal', 'Feasible'):
		gap = relaxation_gap(grid_vars, time_series)
		if gap > RELAXATION_GAP_TOL:
			logger.warning(f'The SOC relaxation of the network is not exact (max v·l - (P² + Q²) = {gap:.2e} pu²): '
						   f'the network results may include fictitious losses; consider increasing '
						   f'grid_data["loss_cost"].')

	return status, opt_value


def relaxation_gap(grid_vars: dict, time_series) -> float:
	"""
	Maximum gap v·l - (P² + Q²) of the SOC relaxation [pu²]; zero (up to tolerances) when the relaxation is exact.
	"""
	gap = 0.0
	for (n, m) in grid_vars['lines']:
		for t in time_series:
			p0, q0 = grid_vars['P'][(n, m)][t].value(), grid_vars['Q'][(n, m)][t].value()
			v0, l0 = grid_vars['v'][n][t].value(), grid_vars['l'][(n, m)][t].value()
			if None not in (p0, q0, v0, l0):
				gap = max(gap, v0 * l0 - (p0 * p0 + q0 * q0))
	return gap


def network_outputs(grid_vars: dict, time_series, n_intervals: int) -> dict:
	"""
	Extracts the network results: branch flows [pu], currents [pu and % of Imax], node voltages [pu]
	and the power exchanged at the slack node [pu].
	"""
	lines, nodes = grid_vars['lines'], grid_vars['nodes']
	outputs = {
		'P_branch': {(n, m): [None] * n_intervals for (n, m) in lines},
		'Q_branch': {(n, m): [None] * n_intervals for (n, m) in lines},
		'I_branch': {(n, m): [None] * n_intervals for (n, m) in lines},
		'I_branch_pct': {(n, m): [None] * n_intervals for (n, m) in lines},
		'v_node': {n: [None] * n_intervals for n in nodes},
		'P_slack': [None] * n_intervals,
		'Q_slack': [None] * n_intervals,
	}
	for (n, m) in lines:
		for t in time_series:
			outputs['P_branch'][(n, m)][t] = grid_vars['P'][(n, m)][t].varValue
			outputs['Q_branch'][(n, m)][t] = grid_vars['Q'][(n, m)][t].varValue
			l_val = grid_vars['l'][(n, m)][t].varValue
			i_val = math.sqrt(max(l_val, 0.0)) if l_val is not None else None
			outputs['I_branch'][(n, m)][t] = i_val
			outputs['I_branch_pct'][(n, m)][t] = i_val / grid_vars['Imax'][(n, m)] * 100 if i_val is not None else None
	for n in nodes:
		for t in time_series:
			v_val = grid_vars['v'][n][t].varValue
			outputs['v_node'][n][t] = math.sqrt(max(v_val, 0.0)) if v_val is not None else None
	for t in time_series:
		outputs['P_slack'][t] = grid_vars['P_slack'][t].varValue
		outputs['Q_slack'][t] = grid_vars['Q_slack'][t].varValue
	return outputs
