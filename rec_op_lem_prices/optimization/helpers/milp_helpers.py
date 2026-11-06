import math
import numpy as np

from rec_op_lem_prices.custom_types.optimization_helpers_types import (
	MetersDict,
	MetersParamDict
)
from typing import Union


def none_lists(m_dims: int) -> list:
	"""
	Creates a list of NaN with len = "m_dims".
	:param m_dims: desired length of the list
	:return: list of NaN
	"""
	master_list = np.empty(m_dims)
	master_list[:] = None
	# Note: it is necessary to convert to list when storing LpVariables.
	return master_list.tolist()


def dict_none_lists(m_dim: int, designations: list[str]) -> dict[list]:
	"""
	Returns a dictionary of NaN lists with len = "m_dim", whose keys are the elements in "designations".
	:param m_dim: desired length of the lists
	:param designations: list of strings that will serve as keys to the returned dictionary
	:return: dictionary pointing to lists of NaN
	"""
	return {d: none_lists(m_dim) for d in designations}


def dict_per_param(meters: MetersDict, values_id: str) -> MetersParamDict:
	"""
	Returns a dictionary with the Meters' IDs as keys and the specified lists of data as values.
	:param meters: data from all Meters
	:param values_id: param key from "meters"
	:return: dictionary whose keys are Meter IDs and the values are the specified lists in values_id
	"""
	return {key: value.get(values_id) for key, value in meters.items()}


def time_intervals(horizon: Union[int, float], delta_t: Union[int, float], func='int') -> int:
	"""
	Retrieves the integer number of time steps within a horizon, provided the duration of those steps.
	By defining "func", the rounding method to integer can be changed.
	:param horizon: an interval of time in hours
	:param delta_t: duration of the time intervals in which th horizon is divided; provided in hours
	:param func: rounding method to achieve an integer; available options are:
	 - "ceil": rounds to the highest nearest integer
	 - "floor": rounds to the smallest nearest integer
	 - "int": truncates the floating point value
	:return: the resulting integer expressing the total number of time intervals
	"""
	if func == 'ceil':
		return math.ceil(horizon / delta_t)
	elif func == 'floor':
		return math.floor(horizon / delta_t)
	elif func == 'int':
		return int(horizon / delta_t)
	else:
		raise ValueError('Please provide a valid division method within the options "ceil", "floor" and "int".')


def sanitize_trip_ev(trip_ev: list, bin_ev: list) -> list:
	"""
	Sets the EV trip energy to zero in every period in which the EV is plugged in at home (bin_ev = 1).
	An EV cannot be travelling while it is plugged in, so any trip energy in those periods is a data
	inconsistency (e.g., produced by the data-extraction heuristic) that would otherwise make the
	problem infeasible whenever trip_ev[t] exceeds what the EV can charge in that period.
	:param trip_ev: EV energy consumption per period [kWh]
	:param bin_ev: 1 if the EV is plugged in at home in that period, 0 otherwise
	:return: the sanitized trip_ev list
	"""
	return [0.0 if bin_ev[t] == 1 else v for t, v in enumerate(trip_ev)]


def round_up(val: float, decimals=3) -> float:
	"""
	Rounds a float to the nearest highest float with the number of decimals specified;
	e.g., for val = 3.12300001 and decimals = 3, the function returns 3.124;
	e.g., for val = 3.12550000 and decimals = 4, the function returns 3.1256.
	Warning: a further increase of the value can happen if the original val has a great number of decimals;
	e.g., for val = 3.12549999999999999 and decimals = 4, the function returns 3.1256.
	:param val: the value to round
	:param decimals: the decimal places that define the approximation
	:return: the rounded value
	"""
	return round(val + 5*10**(-decimals-1), decimals)
