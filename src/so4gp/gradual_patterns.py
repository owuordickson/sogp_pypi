# -*- coding: utf-8 -*-
# SPDX-License-Identifier: GNU GPL v3
# This file is licensed under the terms of the GNU GPL v3.0.
# See the LICENSE file at the root of this
# repository for complete details.

"""
@author: Dickson Owuor
@credits: Thomas Runkler, Edmond Menya, and Anne Laurent
@license: GNU GPL v3
@email: owuordickson@gmail.com
@created: 21 July 2021
@modified: 02 August 2025

A collection of Gradual Pattern classes and methods.
"""
import numpy as np
from dataclasses import dataclass


@dataclass
class PairwiseMatrix:
    bin_mat: np.ndarray
    support: float


class GI:

    def __init__(self, attr_col, symbol):
        """

        GI (Gradual Item). A class that is used to create GI objects. A GI is a pair (i,v) where is a column, and v is a variation symbol -
        increasing/decreasing. Each column of a data set yields 2 GIs; for example, column age yields GI age+ or age-.

        >>> import so4gp as sgp
        >>> gradual_item = sgp.GI(1, "+")
        >>> print(gradual_item.to_string())
        1+

        :param attr_col: Column index
        :type attr_col: int

        :param symbol: Variation symbol either "+" or "-"
        :type symbol: str

        """
        self._attribute_col = attr_col
        """:type attribute_col: int"""
        self._symbol = ""
        """:type symbol: str"""
        if symbol == "-" or symbol == "+":
            self._symbol = symbol
        else:
            raise ValueError("Invalid variation symbol. It should be either '+' or '-'.")

    @property
    def attribute_col(self) -> int:
        """The column index of a GI"""
        return self._attribute_col

    @property
    def symbol(self) -> str:
        """The variation symbol of a GI"""
        return self._symbol

    @property
    def as_tuple(self) -> tuple[int, str]:
        """The Gradual Item (GI) in tuple format"""
        return tuple((self._attribute_col, self._symbol))

    def to_string(self) -> str:
        """Description

        Returns a GI in string format
        :return: string
        """
        return f"{self._attribute_col}{self._symbol}"

    @classmethod
    def from_string(cls, gi_str: str) -> "GI":
        """Creates a GI from a string Gradual Item of the format '1+'"""
        if len(gi_str) != 2:
            raise ValueError("Invalid GI string format. Expected format: '1+' or '1-'")
        
        try:
            attr_col = int(gi_str[0])
            symbol = gi_str[1]
            return cls(attr_col, symbol)
        except ValueError:
            raise ValueError("Invalid attribute column number in GI string")

    @staticmethod
    def swap_gi_symbol(gi_obj: "GI") -> "GI":
        """Description

        Inverts a GI symbol to the opposite variation (i.e., from - to +; or, from + to -)
        :return: inverted GI object
        """
        if gi_obj.symbol == "+":
            sym = "-"
        else:
            sym = "+"
        return GI(gi_obj.attribute_col, sym)

    @staticmethod
    def parse_gi(gi_str: str) -> "GI":
        """

        Converts a stringified GI into normal GI.

        :param gi_str: A stringified GI
        :type gi_str: str

        :return: GI
        """
        txt = gi_str.split('_')
        attr_col = int(txt[0])
        if txt[1] == 'neg':
            symbol = "-"
        else:
            symbol = "+"
        return GI(attr_col, symbol)


class GP:

    def __init__(self):
        """
            GP (Gradual Pattern). A class that is used to create GP objects. A GP object is a set of gradual items (GI), and its quality is
            measured by its computed support value. For example, given a data set with 3 columns (age, salary,
            cars) and 10 objects. A GP may take the form: {age+, salary-} with a support of 0.8. This implies that 8
            out of 10 objects have the values of column age 'increasing' and column 'salary' decreasing.

             The class has the following attributes:
                gradual_items: list if GIs

                support: computed support value as a float
        >>> import so4gp as sgp
        >>> gradual_pattern = sgp.GP()
        >>> gradual_pattern.add_gradual_item(sgp.GI(0, "+"))
        >>> gradual_pattern.add_gradual_item(sgp.GI(1, "-"))
        >>> gradual_pattern.support = 0.5
        >>> print(f"{gradual_pattern.to_string()}: {gradual_pattern.support}")

            """
        self._gradual_items: list[GI] = list()
        self._support: float = 0

    @property
    def gradual_items(self) -> list[GI]:
        return self._gradual_items

    @property
    def support(self) -> float:
        return self._support

    @support.setter
    def support(self, support: float):
        self._support = round(support, 3) if support <= 1 else support

    def add_gradual_item(self, item: GI) -> bool:
        """Description

        Adds a gradual item (GI) into the gradual pattern (GP)
        :param item: gradual item
        :type item: so4gp.GI

        :return: void
        """
        if not isinstance(item, GI):
            raise TypeError("Invalid gradual item")
        self._gradual_items.append(item)

    @property
    def as_set(self) -> set[str]:
        """Returns the gradual pattern (GP) as a set of strings: {'1+', '2-'}"""
        return set(self.to_string())

    @property
    def as_swapped_set(self) -> set[str]:
        """Returns the gradual pattern (GP) as a set of strings: {'1-', '2+'}"""
        gp = GP.swap_gp_symbols(self)
        return set(gp.to_string())

    def decompose(self) -> tuple[list[int], list[str]]:
        """
        Breaks down all the gradual items (GIs) in the gradual pattern into columns and variation symbols and returns
        them as separate variables. For instance, a GP {"1+", "3-"} will be returned as [1, 3], [1, -1]: where [1, 3] is
        the list of attributes/features and [1, -1] are their corresponding gradual variations (1 -> '+' and 1- -> '-').

        :return: Separate columns and variation symbols
        """
        attrs = list()
        syms = list()
        for item in self._gradual_items:
            gi = item.as_tuple
            attrs.append(gi[0])
            syms.append(gi[1])
        return attrs, syms

    def contains_attr(self, gi: GI) -> bool:
        """Description

        Checks is any gradual item (GI) in the gradual pattern (GP) is composed of the column
        :param gi: gradual item
        :type gi: GI

        :return: True if a column exists, False otherwise
        """
        if gi is None:
            return False
        for gi_obj in self._gradual_items:
            if gi.attribute_col == gi_obj.attribute_col:
                return True
        return False

    def to_string(self) -> list[str]:
        """Description

        Returns the GP in string format
        :return: string
        """
        pattern = list()
        for item in self._gradual_items:
            pattern.append(item.to_string())
        return pattern

    def print(self, columns) -> list[list[str] | float]:
        """Description

        A method that returns patterns with actual column names

        :param columns: Column names
        :type columns: list[str]

        :return: GP with actual column names
        """
        pattern = list()
        for item in self._gradual_items:
            col_title = columns[item.attribute_col]
            pat = str(col_title + item.symbol)
            pattern.append(pat)  # (item.to_string())
        return [pattern, self._support]

    def validate_graank(self, d_gp) -> "GP":
        """
        Validates a candidate gradual pattern (GP) based on support computation. A GP is invalid if its support value is
        less than the minimum support threshold set by the user. It uses a breath-first approach to compute support.

        :param d_gp: Data_GP object
        :type d_gp: so4gp.DataGP # noinspection PyTypeChecker

        :return: A valid GP or an empty GP
        """
        # pattern = [('2', "+"), ('4', "+")]
        min_supp = d_gp.thd_supp
        n = d_gp.attr_size
        gi_dict = d_gp.valid_bins.copy()
        gi_key_list = list(gi_dict.keys())

        gen_pattern: GP = GP()
        pw_mat_1: PairwiseMatrix|None = None
        for gi in self.gradual_items:
            arg = np.argwhere(np.isin(np.array(gi_key_list), gi.to_string()))
            if len(arg) > 0:
                i = arg[0][0]
                if pw_mat_1 is None:
                    pw_mat_1 = gi_dict[gi_key_list[i]]
                    gen_pattern.add_gradual_item(gi)
                else:
                    pw_mat_2 = gi_dict[gi_key_list[i]]
                    res_pw_mat = GP.perform_and(pw_mat_1, pw_mat_2, n)
                    if res_pw_mat.support >= min_supp:
                        pw_mat_1 = PairwiseMatrix(bin_mat=res_pw_mat.bin_mat.copy(), support=res_pw_mat.support)
                        gen_pattern.add_gradual_item(gi)
                        gen_pattern.support = res_pw_mat.support
        if len(gen_pattern.gradual_items) <= 1:
            return self
        else:
            return gen_pattern

    def validate_tree(self, d_gp):
        """
        Validates a candidate gradual pattern (GP) based on support computation. A GP is invalid if its support value is
        less than the minimum support threshold set by the user. It applies a depth-first (FP-Growth) approach
        to compute support.

        :param d_gp: Data_GP object
        :type d_gp: so4gp.DataGP # noinspection PyTypeChecker

        :return: A valid GP or an empty GP
        """
        if d_gp.valid_tids is None:
            return self

        min_supp = d_gp.thd_supp
        n = d_gp.row_count
        gen_pattern = GP()
        """type gen_pattern: GP"""
        temp_tids = None
        for gi in self.gradual_items:
            node = gi.to_string()
            node_inv = GI.swap_gi_symbol(gi).to_string()
            for gi_str, gi_tids in d_gp.valid_tids.items():
                if (node == gi_str) or (node_inv == gi_str):
                    if temp_tids is None:
                        temp_tids = gi_tids
                        gen_pattern.add_gradual_item(gi)
                    else:
                        temp = temp_tids.copy()
                        temp = temp.intersection(gi_tids)
                        supp = float(len(temp)) / float(n * (n - 1.0) / 2.0)
                        if supp >= min_supp:
                            temp_tids = temp.copy()
                            gen_pattern.add_gradual_item(gi)
                            gen_pattern.support = supp
        if len(gen_pattern.gradual_items) <= 1:
            return self
        else:
            return gen_pattern

    def check_am(self, gp_list: list["GP"], subset:bool=True) -> bool:
        """
        Anti-monotonicity check. Checks if a GP is a subset or superset of an already existing GP

        :param gp_list: A list of existing GPs
        :param subset: A check if it is a subset
        :return: True if superset/subset, False otherwise
        """
        result = False
        if subset:
            for pat in gp_list:
                result1 = set(self.as_set).issubset(set(pat.as_set))
                result2 = set(self.as_swapped_set).issubset(set(pat.as_set))
                if result1 or result2:
                    result = True
                    break
        else:
            for pat in gp_list:
                result1 = set(self.as_set).issuperset(set(pat.as_set))
                result2 = set(self.as_swapped_set).issuperset(set(pat.as_set))
                if result1 or result2:
                    result = True
                    break
        return result

    def is_duplicate(self, valid_gps:list["GP"], invalid_gps:list["GP"]=None) -> bool:
        """Description

        Checks if a pattern is in the list of winner GPs or loser GPs

        :param valid_gps: list of GPs
        :param invalid_gps: list of GPs
        :return: True if a pattern is a list, False otherwise
        """
        if invalid_gps is None:
            pass
        else:
            for pat in invalid_gps:
                if set(self.as_set) == set(pat.as_set) or \
                        set(self.as_swapped_set) == set(pat.as_set):
                    return True
        for pat in valid_gps:
            if set(self.as_set) == set(pat.as_set) or \
                    set(self.as_swapped_set) == set(pat.as_set):
                return True
        return False

    @staticmethod
    def swap_gp_symbols(gp_obj: "GP") -> "GP":
        """
        Swaps the variation symbols of all the gradual items (GIs) in a gradual pattern (GP)
        """
        new_gp = GP()
        for gi in gp_obj.gradual_items:
            new_gp.add_gradual_item(GI.swap_gi_symbol(gi))
        return new_gp

    @staticmethod
    def perform_and(bin_data_1: "PairwiseMatrix", bin_data_2: "PairwiseMatrix", dim: int) -> "PairwiseMatrix":
        """
        Perform logical AND operation on two bitmaps.
        :param bin_data_1: Bitmap 1
        :param bin_data_2: bitmap 2
        :param dim: dimension of the bitmaps
        """
        bin_mat = bin_data_1.bin_mat * bin_data_2.bin_mat
        sup = float(np.sum(bin_mat)) / float(dim * (dim - 1.0) / 2.0)
        return PairwiseMatrix(bin_mat=bin_mat, support=sup)


class TimeDelay:

    def __init__(self, tstamp=0, supp=0):
        """
            TimeDelay (Time Delay). A class used in Fuzzy Temporal Gradual Patterns to create the time-delay object. The class TimeDelay has the following attributes:

            timestamp: the time-delay value as a timestamp.

            support: the true value of the time-delay value.

            valid: if the time-delay value is valid (should not be zero).

            sign: if the time is earlier (-) or later (+)

            formatted_time: time-delay formatted as a Date (in terms of hours/days/weeks/months/years)

        >>> import so4gp as sgp
        >>> t_delay = sgp.TimeDelay(3600, 0.75)
        >>> t_delay.to_string()

        :param tstamp: The time-delay value as a timestamp.
        :type tstamp: Float

        :param supp: The true value of the time-delay value.
        :type supp: Float
        """
        self._timestamp: float = tstamp
        self._support: float = round(supp, 3)
        self._valid: bool = False
        self._sign: str = ""
        self._formatted_time: dict = {}
        self._init_parameters()

    @property
    def timestamp(self) -> float:
        return self._timestamp

    @property
    def support(self) -> float:
        return self._support

    @property
    def valid(self) -> bool:
        return self._valid

    @property
    def sign(self) -> str:
        return self._sign

    @property
    def formatted_time(self) -> dict:
        return self._formatted_time

    def _init_parameters(self):
        """Initializes the class parameters."""

        def delay_sign() -> str:
            """
            Checks and returns the sign of the time-delay value (later/before).

            :return: The sign of the time-delay value.
            """
            if self._timestamp < 0:
                return "-"
            else:
                return "+"

        def format_time() -> list:
            """
            Formats the time-delay value as a Date in string format (i.e., seconds/minutes/hours/days/weeks/months/years).

            :return: The formatted time-delay as a list.
            """
            stamp_in_seconds = abs(self._timestamp)
            years = stamp_in_seconds / 3.154e+7
            months = stamp_in_seconds / 2.628e+6
            weeks = stamp_in_seconds / 604800
            days = stamp_in_seconds / 86400
            hours = stamp_in_seconds / 3600
            minutes = stamp_in_seconds / 60
            if int(years) <= 0:
                if int(months) <= 0:
                    if int(weeks) <= 0:
                        if int(days) <= 0:
                            if int(hours) <= 0:
                                if int(minutes) <= 0:
                                    return [round(stamp_in_seconds, 0), "seconds"]
                                else:
                                    return [round(minutes, 0), "minutes"]
                            else:
                                return [round(hours, 0), "hours"]
                        else:
                            return [round(days, 0), "days"]
                    else:
                        return [round(weeks, 0), "weeks"]
                else:
                    return [round(months, 0), "months"]
            else:
                return [round(years, 0), "years"]

        self._sign: str = delay_sign()
        self._formatted_time: dict = {}
        if self._timestamp != 0:
            time_arr = format_time()
            self._formatted_time = {'value': time_arr[0], 'duration': time_arr[1]}
            self._valid = True

    def to_string(self) -> str:
        """
        Returns formated time-delay as a string.

        :return: The time-delay as a string.
        """
        if not self._formatted_time:
            txt = ("~ " + self._sign + str(self._formatted_time['value']) + " " + str(self._formatted_time['duration'])
                   + " : " + str(self._support))
        else:
            txt = "No time lag found!"
        return txt


class TGP(GP):

    @dataclass
    class TemporalGI:
        gradual_item: GI
        time_delay: TimeDelay

    def __init__(self):
        """
        Description of class TGP (Temporal Gradual Pattern)

        A class that inherits an existing GP class to create Temporal GP objects. A TGP is a gradual pattern with a
        time-delay. It has a target gradual item (which is created from a user-defined attribute), and it is used as the
        anchor for mining patterns from a dataset. The class has the following attributes:

        target_gradual_item: the gradual item on which the pattern is based.

        temporal_gradual_items: gradual items which occur after specific time delays.

        >>> import so4gp as sgp
        >>> t_gp = sgp.TGP()
        >>> t_gp.target_gradual_item = sgp.GI(1, "+")
        >>> t_gp.temporal_gradual_items(sgp.GI(2, "-"), sgp.TimeDelay(7200, 0.8))
        >>> t_gp.to_string()
        """
        super(TGP, self).__init__()
        self._target_gradual_item: GI|None = None
        self._temporal_gradual_items: list[TGP.TemporalGI] = list()

    @property
    def target_gradual_item(self) -> GI:
        return self._target_gradual_item

    @target_gradual_item.setter
    def target_gradual_item(self, item: GI) -> None:
        """Adds a target gradual item (fTGI) into the fuzzy temporal gradual pattern (fTGP)"""
        if not isinstance(item, GI):
            raise TypeError("Target gradual item must be of type GI")
        self._target_gradual_item = item

    def add_temporal_gradual_item(self, item: GI, time_delay: TimeDelay):
        """Description

            Adds a fuzzy temporal gradual item (fTGI) into the fuzzy temporal gradual pattern (fTGP)
            :param item: gradual item
            :type item: so4gp.GI

            :param time_delay: time delay
            :type time_delay: TimeDelay

            :return: void
        """
        if isinstance(item, GI) and isinstance(time_delay, TimeDelay):
            temp_gi = TGP.TemporalGI(gradual_item=item, time_delay=time_delay)
            self._temporal_gradual_items.append(temp_gi)
        else:
            raise TypeError("Invalid arguments - require GI and TimeDelay objects")

    def to_string(self) -> list:
        """
        Returns the Temporal-GP in string format as a list.
        """
        pattern = [self._target_gradual_item.to_string()]
        for temp_gi in self._temporal_gradual_items:
            gi = temp_gi.gradual_item
            t_lag = temp_gi.time_delay
            str_time = f"{t_lag.sign}{t_lag.formatted_time['value']} {t_lag.formatted_time['duration']}"
            pattern.append(f"({gi.to_string()}) {str_time}")
        return pattern

    def print(self, columns) -> list[list[str] | float]:
        """Description

        A method that returns a fuzzy temporal gradual pattern (TGP) with actual column names

        :param columns: Column names
        :type columns: list[str]

        :return: TGP with actual column names
        """

        target_gi = self._target_gradual_item
        col_title = columns[target_gi.attribute_col]
        pattern = [f"{col_title}{target_gi.symbol}"]

        for temp_gi in self._temporal_gradual_items:
            gi = temp_gi.gradual_item
            t_lag = temp_gi.time_delay
            str_time = f"{t_lag.sign}{t_lag.formatted_time['value']} {t_lag.formatted_time['duration']}"
            col_title = columns[gi.attribute_col]
            pat = f"({col_title}{gi.symbol}) {str_time}"
            pattern.append(pat)
        return [pattern, self.support]