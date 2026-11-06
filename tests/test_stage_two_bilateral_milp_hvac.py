import copy
import math

import pytest

from rec_op_lem_prices.optimization.module.StageTwoMILPBilateral import StageTwoMILPBilateral
from rec_op_lem_prices.optimization.structures.I_O_stage_2_bilateral_milp_hvac import (
	INPUTS_S2_BILATERAL_HVAC,
	OUTPUTS_S2_BILATERAL_HVAC
)

import matplotlib.pyplot as plt

def test_solve_collective_bilateral_milp():
	# Assert the creation of a correct class
	milp = StageTwoMILPBilateral(INPUTS_S2_BILATERAL_HVAC)
	assert isinstance(milp, StageTwoMILPBilateral)

	# Assert the MILP is optimally solved
	milp.solve_milp()
	assert milp.status == 'Optimal'
	# Assert the correct outputs
	results = milp.generate_outputs()
	round_cost = lambda x: {meter_id: round(cost, 3) for meter_id, cost in x.items()}
	results['obj_value'] = round(results['obj_value'], 3)
	results['c_ind2bilateral'] = round_cost(results['c_ind2bilateral'])
	results['c_ind2bilateral_without_deg'] = round_cost(results['c_ind2bilateral_without_deg'])
	results['c_ind2bilateral_without_deg_and_p_extra'] = round_cost(results['c_ind2bilateral_without_deg_and_p_extra'])
	results['c_ind2bilateral_without_p_extra'] = round_cost(results['c_ind2bilateral_without_p_extra'])
	for ki, valu in results.items():
		assert valu == OUTPUTS_S2_BILATERAL_HVAC.get(ki), f'{ki}'


@pytest.mark.parametrize('t_out, allowed_sign', [
	(8.0, +1),   # cold outside: the 'state' HVAC may only be off or heating
	(32.0, -1),  # hot outside: the 'state' HVAC may only be off or cooling
])
def test_state_hvac_follows_thermal_dynamics(t_out, allowed_sign):
	# Same inputs as above, but with a constant outdoor temperature for the 'state' HVAC (Meter#1 / HVAC#2)
	inputs = copy.deepcopy(INPUTS_S2_BILATERAL_HVAC)
	hvac = inputs['meters']['Meter#1']['hvac']['HVAC#2']
	assert hvac['type'] == 'state'
	hvac['t_out'] = [t_out] * len(hvac['t_out'])

	milp = StageTwoMILPBilateral(inputs)
	milp.solve_milp()
	assert milp.status == 'Optimal'
	results = milp.generate_outputs()

	temp = results['hvac_temp']['Meter#1']['HVAC#2']
	power = results['hvac_power']['Meter#1']['HVAC#2']
	r, c = hvac['thermal_resist'], hvac['thermal_cap']
	decay = math.exp(-inputs['delta_t'] / (r * c))

	assert temp[0] == pytest.approx(hvac['init_temp'])
	for t in range(1, len(temp)):
		# T[t] = T_inf - (T_inf - T[t-1]) * exp(-dt / RC), with T_inf = T_out (+/-) P * R
		t_inf = t_out + allowed_sign * power[t] * r
		expected = t_inf - (t_inf - temp[t - 1]) * decay
		assert temp[t] == pytest.approx(expected, abs=1e-3), f't={t}'


if __name__ == '__main__':
	test_solve_collective_bilateral_milp()
