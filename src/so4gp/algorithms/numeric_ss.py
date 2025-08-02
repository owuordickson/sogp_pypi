# -*- coding: utf-8 -*-
# SPDX-License-Identifier: GNU GPL v3
# This file is licensed under the terms of the GNU GPL v3.0.
# See the LICENSE file at the root of this
# repository for complete details.


import numpy as np
from ypstruct import structure

try:
    from . import DataGP, GI, ExtGP
except ImportError:
    from src.so4gp import DataGP, GI, ExtGP

class NumericSS:
    """Description of class NumericSS (Numeric Search Space)

    A class that implements functions that allow swarm algorithms to explore a numeric search space.

    The class NumericSS has the following functions:
        decode_gp: decodes a GP from a numeric position
        cost_function: computes the fitness of a GP
        apply_bound: applies minimum and maximum values

    """

    def __init__(self):
        pass

    @staticmethod
    def decode_gp(attr_keys: list, position: float) -> ExtGP:
        """Description

        Decodes a numeric value (position) into a GP

        :param attr_keys: list of attribute keys
        :param position: a value in the numeric search space
        :return: GP that is decoded from the position value
        """

        temp_gp = ExtGP()
        ":type temp_gp: ExtGP"
        if position is None:
            return temp_gp

        bin_str = bin(int(position))[2:]
        bin_arr = np.array(list(bin_str), dtype=int)

        for i in range(bin_arr.size):
            bin_val = bin_arr[i]
            if bin_val == 1:
                gi = GI.parse_gi(attr_keys[i])
                if not temp_gp.contains_attr(gi):
                    temp_gp.add_gradual_item(gi)
        return temp_gp

    @staticmethod
    def cost_function(position: float, attr_keys: list, d_set: DataGP) -> float:
        """Description

        Computes the fitness of a GP

        :param position: a value in the numeric search space
        :param attr_keys: list of attribute keys
        :param d_set: a DataGP object
        :return: a floating point value that represents the fitness of the position
        """

        pattern = NumericSS.decode_gp(attr_keys, position)
        temp_bin = np.array([])
        for gi in pattern.gradual_items:
            arg = np.argwhere(np.isin(d_set.valid_bins[:, 0], gi.as_array))
            if len(arg) > 0:
                i = arg[0][0]
                valid_bin = d_set.valid_bins[i]
                if temp_bin.size <= 0:
                    temp_bin = valid_bin[1].copy()
                else:
                    temp_bin = np.multiply(temp_bin, valid_bin[1])
        bin_sum = np.sum(temp_bin)
        if bin_sum > 0:
            cost = (1 / bin_sum)
        else:
            cost = 1
        return cost

    @staticmethod
    def apply_bound(x: structure, var_min: int, var_max: int) -> None:
        """
        Modifies x (a numeric value) if it exceeds the lower/upper bound of the numeric search space.

        :param x: A value in the numeric search space
        :param var_min: The lower-bound value
        :param var_max: The upper-bound value
        :return: None
        """

        x.position = np.maximum(x.position, var_min)
        x.position = np.minimum(x.position, var_max)
