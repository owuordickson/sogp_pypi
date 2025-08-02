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
        self.attribute_col = attr_col
        """:type attribute_col: int"""
        self.symbol = symbol
        """:type symbol: str"""
        self.rank_sum: int = 0

    @property
    def is_decreasing(self) -> bool:
        """Checks if a GI's variation corresponds to decreasing and returns True, False otherwise."""
        if self.symbol == "-":
            return True
        else:
            return False

    @property
    def as_array(self) -> np.ndarray:
        """The Gradual Item (GI) in ndarray format"""
        return np.array((self.attribute_col, self.symbol), dtype='i, S1')

    @property
    def as_swapped_array(self) -> np.ndarray:
        """The resulting GI with symbol swapped (i.e., from - to +; or, from + to -) in ndarray format"""
        if self.symbol == "+":
            # tuple([self.attribute_col, "-"])
            return np.array((self.attribute_col, "-"), dtype='i, S1')
        elif self.symbol == "-":
            # tuple([self.attribute_col, "+"])
            return np.array((self.attribute_col, "+"), dtype='i, S1')
        else:
            return np.array((self.attribute_col, 'x'), dtype='i, S1')

    @property
    def as_tuple(self) -> tuple[int, str]:
        """The Gradual Item (GI) in tuple format"""
        return tuple((self.attribute_col, self.symbol))

    @property
    def as_integer(self) -> list[int]:
        """The Gradual Item (GI) in integer format. Converts a variation symbol into an integer
        (i.e., + to 1; and - to -1) and returns as a list (e.g., (1+) -> [1, -1])."""
        if self.symbol == "+":
            return [self.attribute_col, 1]
        elif self.symbol == "-":
            return [self.attribute_col, -1]
        else:
            return [self.attribute_col, 0]

    @property
    def as_string(self) -> str:
        """The Gradual Item (GI) in string format. Stringifies a GI. It converts a variation symbol into a string
        (i.e., + to _pos; and - to _neg)."""
        if self.symbol == "+":
            return str(self.attribute_col) + "_pos"
        elif self.symbol == "-":
            return str(self.attribute_col) + "_neg"
        else:
            return str(self.attribute_col) + "_inv"

    def to_string(self) -> str:
        """Description

        Returns a GI in string format
        :return: string
        """
        return str(self.attribute_col) + self.symbol

    @staticmethod
    def swap_gi_symbol(gi_obj):
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
    def parse_gi(gi_str):
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
        >>> gradual_pattern.set_support(0.5)
        >>> print(f"{gradual_pattern.to_string()}: {gradual_pattern.support}")

            """
        self.gradual_items = list()
        """:type gradual_items: list[GI] """
        self.support = 0
        """:type support: float"""

    def set_support(self, support) -> bool:
        """Description

        Sets the computed support value of the gradual pattern (GP)
        :param support: support value
        :type support: float

        :return: void
        """
        if support <= 1:
            self.support = round(support, 3)
            return True
        return False

    def add_gradual_item(self, item) -> bool:
        """Description

        Adds a gradual item (GI) into the gradual pattern (GP)
        :param item: gradual item
        :type item: so4gp.GI

        :return: void
        """
        if item.symbol == "-" or item.symbol == "+":
            self.gradual_items.append(item)
            return True
        return False

    def add_items_from_list(self, lst_items) -> None:
        """
        Adds gradual items from a list of str or a list of sets.
        For example,
        >>> import so4gp
        >>> new_gp = so4gp.GP()
        >>> new_gp.add_items_from_list(["0+", "2-", "3-"])

        :param lst_items: A string or set
        :type lst_items: list
        """
        for str_gi in lst_items:
            if type(str_gi[1]) is str:
                self.add_gradual_item(GI(int(str_gi[0]), str_gi[1]))
            elif type(str_gi[1]) is bytes:
                self.add_gradual_item(GI(int(str_gi[0]), str(str_gi[1].decode())))

    @property
    def as_list(self) -> list:
        """Description

        Returns the gradual pattern (GP) as a list
        :return: gradual pattern
        """
        pattern = list()
        for gi in self.gradual_items:
            pattern.append(gi.as_array.tolist())
        return pattern

    @property
    def as_swapped_list(self) -> list:
        """Description

        Swaps all the variation symbols of all the gradual items (GIs) in the gradual pattern (GP)
        :return: GP with swapped variation symbols
        """
        pattern = list()
        for gi in self.gradual_items:
            pattern.append(gi.as_swapped_array.tolist())
        return pattern

    @property
    def as_array(self) -> np.ndarray:
        """Description

        Returns a gradual pattern (GP) as a ndarray
        :return: ndarray
        """
        pattern = []
        for gi in self.gradual_items:
            pattern.append(gi.as_array)
        return np.array(pattern)

    @property
    def as_tuples(self) -> list[tuple[int, str]]:
        """Description

        Returns the gradual pattern (GP) as a list of GI tuples
        :return: list of GI tuples
        """
        pattern = list()
        for gi in self.gradual_items:
            pattern.append(gi.as_tuple)
        return pattern

    def decompose(self) -> tuple[list[int], list[str]]:
        """
        Breaks down all the gradual items (GIs) in the gradual pattern into columns and variation symbols and returns
        them as separate variables. For instance, a GP {"1+", "3-"} will be returned as [1, 3], [1, -1]: where [1, 3] is
        the list of attributes/features and [1, -1] are their corresponding gradual variations (1 -> '+' and 1- -> '-').

        :return: Separate columns and variation symbols
        """
        attrs = list()
        syms = list()
        for item in self.gradual_items:
            gi = item.as_integer
            attrs.append(gi[0])
            syms.append(gi[1])
        return attrs, syms

    def find_index(self, gi) -> int:
        """Description

        Returns the index position of a gradual item in the gradual pattern
        :param gi: gradual item
        :type gi: GI

        :return: index of gradual item
        """
        for i in range(len(self.gradual_items)):
            gi_obj = self.gradual_items[i]
            if (gi.symbol == gi_obj.symbol) and (gi.attribute_col == gi_obj.attribute_col):
                return i
        return -1

    def contains(self, gi) -> bool:
        """Description

        Checks if a gradual item (GI) is a member of a gradual pattern (GP)
        :param gi: gradual item
        :type gi: GI

        :return: True if it is a member, otherwise False
        """
        if gi is None:
            return False
        if gi in self.gradual_items:
            return True
        return False

    def contains_strict(self, gi) -> bool:
        """Description

        Strictly checks if a gradual item (GI) is a member of a gradual pattern (GP)
        :param gi: gradual item
        :type gi: GI

        :return: True if it is a member, otherwise False
        """
        if gi is None:
            return False
        for gi_obj in self.gradual_items:
            if (gi.attribute_col == gi_obj.attribute_col) and (gi.symbol == gi_obj.symbol):
                return True
        return False

    def contains_attr(self, gi) -> bool:
        """Description

        Checks is any gradual item (GI) in the gradual pattern (GP) is composed of the column
        :param gi: gradual item
        :type gi: GI

        :return: True if a column exists, False otherwise
        """
        if gi is None:
            return False
        for gi_obj in self.gradual_items:
            if gi.attribute_col == gi_obj.attribute_col:
                return True
        return False

    def to_string(self) -> list[str]:
        """Description

        Returns the GP in string format
        :return: string
        """
        pattern = list()
        for item in self.gradual_items:
            pattern.append(item.to_string())
        return pattern

    def to_dict(self) -> dict[str, int]:
        """Description

        Returns the GP as a dictionary
        :return: dict
        """
        gi_dict = {}
        for gi in self.gradual_items:
            gi_dict.update({gi.as_string: 0})
        return gi_dict

    # noinspection PyUnresolvedReferences
    def print(self, columns) -> list[list[str] | float]:
        """Description

        A method that returns patterns with actual column names

        :param columns: Column names
        :type columns: list[str]

        :return: GP with actual column names
        """
        pattern = list()
        for item in self.gradual_items:
            col_title = columns[item.attribute_col]
            try:
                col = str(col_title.value.decode())
            except AttributeError:
                col = str(col_title[1].decode())
            pat = str(col + item.symbol)
            pattern.append(pat)  # (item.to_string())
        return [pattern, self.support]


class ExtGP(GP):

    def __init__(self):
        """Description of class ExtGP (Extended Gradual Pattern)

        A class that inherits class GP which is used to create more powerful GP objects that can be used in mining
        approaches that implement swarm optimization techniques or cluster analysis or classification algorithms.

        It adds the following attribute:
            freq_count: frequency count of a particular GP object.

        >>> import so4gp as sgp
        >>> gradual_pattern = sgp.ExtGP()
        >>> gradual_pattern.add_gradual_item(sgp.GI(0, "+"))
        >>> gradual_pattern.add_gradual_item(sgp.GI(1, "-"))
        >>> gradual_pattern.set_support(0.5)
        >>> print(f"{gradual_pattern.to_string()}: {gradual_pattern.support}")

        """
        super(ExtGP, self).__init__()
        self.freq_count = 0
        """:type freq_count: int"""

    def validate_graank(self, d_gp):
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
        gen_pattern = ExtGP()
        """type gen_pattern: ExtGP"""
        bin_arr = np.array([])

        for gi in self.gradual_items:
            arg = np.argwhere(np.isin(d_gp.valid_bins[:, 0], gi.as_array))
            if len(arg) > 0:
                i = arg[0][0]
                valid_bin = d_gp.valid_bins[i]
                if bin_arr.size <= 0:
                    bin_arr = np.array([valid_bin[1], valid_bin[1]])
                    gen_pattern.add_gradual_item(gi)
                else:
                    bin_arr[1] = valid_bin[1].copy()
                    temp_bin = np.multiply(bin_arr[0], bin_arr[1])
                    supp = float(np.sum(temp_bin)) / float(n * (n - 1.0) / 2.0)
                    if supp >= min_supp:
                        bin_arr[0] = temp_bin.copy()
                        gen_pattern.add_gradual_item(gi)
                        gen_pattern.set_support(supp)
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
        min_supp = d_gp.thd_supp
        n = d_gp.row_count
        gen_pattern = ExtGP()
        """type gen_pattern: ExtGP"""
        temp_tids = None
        for gi in self.gradual_items:
            gi_int = gi.as_integer
            node = int(gi_int[0] + 1) * gi_int[1]
            gi_int = (GI.swap_gi_symbol(gi)).as_integer
            node_inv = int(gi_int[0] + 1) * gi_int[1]
            for k, v in d_gp.valid_tids.items():
                if (node == k) or (node_inv == k):
                    if temp_tids is None:
                        temp_tids = v
                        gen_pattern.add_gradual_item(gi)
                    else:
                        temp = temp_tids.copy()
                        temp = temp.intersection(v)
                        supp = float(len(temp)) / float(n * (n - 1.0) / 2.0)
                        if supp >= min_supp:
                            temp_tids = temp.copy()
                            gen_pattern.add_gradual_item(gi)
                            gen_pattern.set_support(supp)
                """elif node_inv == k:
                    if temp_tids is None:
                        temp_tids = v
                        gen_pattern.add_gradual_item(gi)
                    else:
                        temp = temp_tids.copy()
                        temp = temp.intersection(v)
                        supp = float(len(temp)) / float(n * (n - 1.0) / 2.0)
                        if supp >= min_supp:
                            temp_tids = temp.copy()
                            gen_pattern.add_gradual_item(gi)
                            gen_pattern.set_support(supp)"""
        if len(gen_pattern.gradual_items) <= 1:
            return self
        else:
            return gen_pattern

    def check_am(self, gp_list, subset=True) -> bool:
        """
        Anti-monotonicity check. Checks if a GP is a subset or superset of an already existing GP

        :param gp_list: A list of existing GPs
        :type gp_list: list[so4gp.ExtGP]

        :param subset: A check if it is a subset
        :type subset: bool

        :return: True if superset/subset, False otherwise
        """
        result = False
        if subset:
            for pat in gp_list:
                result1 = set(self.as_list).issubset(set(pat.get_pattern()))
                result2 = set(self.as_swapped_list).issubset(set(pat.get_pattern()))
                if result1 or result2:
                    result = True
                    break
        else:
            for pat in gp_list:
                result1 = set(self.as_list).issuperset(set(pat.get_pattern()))
                result2 = set(self.as_swapped_list).issuperset(set(pat.get_pattern()))
                if result1 or result2:
                    result = True
                    break
        return result

    def is_duplicate(self, valid_gps, invalid_gps=None) -> bool:
        """Description

        Checks if a pattern is in the list of winner GPs or loser GPs

        :param valid_gps: list of GPs
        :type valid_gps: list[so4gp.ExtGP]

        :param invalid_gps: list of GPs
        :type invalid_gps: list[so4gp.ExtGP]

        :return: True if a pattern is either list, False otherwise
        """
        if invalid_gps is None:
            pass
        else:
            for pat in invalid_gps:
                if set(self.as_list) == set(pat.get_pattern()) or \
                        set(self.as_swapped_list) == set(pat.get_pattern()):
                    return True
        for pat in valid_gps:
            if set(self.as_list) == set(pat.get_pattern()) or \
                    set(self.as_swapped_list) == set(pat.get_pattern()):
                return True
        return False

    @staticmethod
    def remove_subsets(gp_list, gi_arr) -> list:
        """

        Remove subset GPs from the list.

        :param gp_list: List of existing GPs
        :type gp_list: list[so4gp.ExtGP]

        :param gi_arr: Gradual items in an array
        :type gi_arr: set

        :return: List of GPs
        """
        mod_gp_list = []
        for gp in gp_list:
            result1 = set(gp.get_pattern()).issubset(gi_arr)
            result2 = set(gp.inv_pattern()).issubset(gi_arr)
            if not (result1 or result2):
                mod_gp_list.append(gp)

        return mod_gp_list


class TGP(ExtGP):

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
        >>> t_gp.add_target_gradual_item(sgp.GI(1, "+"))
        >>> t_gp.temporal_gradual_items(sgp.GI(2, "-"), sgp.TimeDelay(7200, 0.8))
        >>> t_gp.to_string()
        """
        super(TGP, self).__init__()
        self.target_gradual_item = GI(-1, "")
        """:type target_gradual_item: GI"""
        self.temporal_gradual_items = list()
        """:type temporal_gradual_items: list()"""

    def add_target_gradual_item(self, item) -> bool:
        """Description

            Adds a target gradual item (fTGI) into the fuzzy temporal gradual pattern (fTGP)
            :param item: gradual item
            :type item: so4gp.GI

            :return: void
        """
        if item.symbol == "-" or item.symbol == "+":
            self.gradual_items.append(item)
            self.target_gradual_item = item
            return True
        return False

    def add_temporal_gradual_item(self, item, time_delay) -> bool:
        """Description

            Adds a fuzzy temporal gradual item (fTGI) into the fuzzy temporal gradual pattern (fTGP)
            :param item: gradual item
            :type item: so4gp.GI

            :param time_delay: time delay
            :type time_delay: TimeDelay

            :return: void
        """
        if item.symbol == "-" or item.symbol == "+":
            self.gradual_items.append(item)
            self.temporal_gradual_items.append([item, time_delay])
            return True
        return False

    def to_string(self) -> list:
        """
        Returns the Temporal-GP in string format as a list.
        """
        pattern = [self.target_gradual_item.to_string()]
        for item, t_lag in self.temporal_gradual_items:
            str_time = f"{t_lag.sign}{t_lag.formatted_time['value']} {t_lag.formatted_time['duration']}"
            pattern.append(f"({item.to_string()}) {str_time}")
        return pattern

    # noinspection PyUnresolvedReferences
    def print(self, columns) -> list[list[str] | float]:
        """Description

        A method that returns a fuzzy temporal gradual pattern (TGP) with actual column names

        :param columns: Column names
        :type columns: list[str]

        :return: TGP with actual column names
        """

        target_gi = self.target_gradual_item
        col_title = columns[target_gi.attribute_col]
        try:
            col = str(col_title.value.decode())
        except AttributeError:
            col = str(col_title[1].decode())
        pattern = [f"{col}{target_gi.symbol}"]

        for item, t_lag in self.temporal_gradual_items:
            str_time = f"{t_lag.sign}{t_lag.formatted_time['value']} {t_lag.formatted_time['duration']}"
            col_title = columns[item.attribute_col]
            try:
                col = str(col_title.value.decode())
            except AttributeError:
                col = str(col_title[1].decode())
            pat = f"({col}{item.symbol}) {str_time}"
            pattern.append(pat)
        return [pattern, self.support]


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
        self.timestamp = tstamp
        """type: timestamp: float"""
        self.support = round(supp, 3)
        """:type support: float"""
        self.valid = False
        """type: valid: bool"""
        self.sign = self.delay_sign
        """type: sign: str"""
        self.formatted_time = {}
        """type: formatted_time: dict"""
        if tstamp != 0:
            time_arr = self._format_time()
            self.formatted_time = {'value': time_arr[0], 'duration': time_arr[1]}
            self.valid = True

    @property
    def delay_sign(self) -> str:
        """
        Checks and returns the sign of the time-delay value (later/before).

        :return: The sign of the time-delay value.
        """
        if self.timestamp < 0:
            return "-"
        else:
            return "+"

    def _format_time(self) -> list:
        """
        Formats the time-delay value as a Date in string format (i.e., seconds/minutes/hours/days/weeks/months/years).

        :return: The formatted time-delay as a list.
        """
        stamp_in_seconds = abs(self.timestamp)
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

    def to_string(self) -> str:
        """
        Returns formated time-delay as a string.

        :return: The time-delay as a string.
        """
        if not self.formatted_time:
            txt = ("~ " + self.sign + str(self.formatted_time['value']) + " " + str(self.formatted_time['duration'])
                   + " : " + str(self.support))
        else:
            txt = "No time lag found!"
        return txt
