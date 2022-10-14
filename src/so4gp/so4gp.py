# -*- coding: utf-8 -*-
"""
@author: Dickson Owuor
@credits: Thomas Runkler, Edmond Menya, and Anne Laurent
@license: MIT
@version: 0.3.4
@email: owuordickson@gmail.com
@created: 21 July 2021
@modified: 14 October 2022

SO4GP
------

    **SO4GP** stands for: "Some Optimizations for Gradual Patterns". SO4GP applies optimizations such as swarm
    intelligence, HDF5 chunks, SVD and many others in order to improve the efficiency of extracting gradual patterns
    (GPs). A GP is a set of gradual items (GI) and its quality is measured by its computed support value. A GI is a pair
    (i,v) where i is a column and v is a variation symbol: increasing/decreasing. Each column of a data set yields 2
    GIs; for example, column age yields GI age+ or age-. For example given a data set with 3 columns (age, salary, cars)
    and 10 objects. A GP may take the form: {age+, salary-} with a support of 0.8. This implies that 8 out of 10 objects
    have the values of column age 'increasing' and column 'salary' decreasing.

    The classical approach for mining GPs is computationally expensive. This package provides Python algorithm
    implementations of several optimization techniques that are applied to the classical approach in order to improve
    its computational efficiency. The algorithm implementations include:
        * (Classical) GRAANK algorithm for extracting GPs
        * Ant Colony Optimization algorithm for extracting GPs
        * Genetic Algorithm for extracting GPs
        * Particle Swarm Optimization algorithm for extracting GPs
        * Random Search algorithm for extracting GPs
        * Local Search algorithm for extracting GPs

    Apart from swarm-based optimization techniques, this package also provides a Python algorithm implementation of a
    clustering approach for mining GPs.

"""

import csv
from collections import defaultdict
from dateutil.parser import parse
import time
import gc
import math
import numpy as np
import json
import multiprocessing as mp
import os
import pandas as pd
import random
import statistics
from tabulate import tabulate
from ypstruct import structure
from sklearn.cluster import KMeans


# -------- CONFIGURATION -------------

# Global Swarm Configurations
MIN_SUPPORT = 0.5
MAX_ITERATIONS = 1
N_VAR = 1  # DO NOT CHANGE

# ACO-GRAANK Configurations:
EVAPORATION_FACTOR = 0.5

# GA-GRAANK Configurations:
N_POPULATION = 5
PC = 0.5
GAMMA = 1  # Cross-over
MU = 0.9  # Mutation
SIGMA = 0.9  # Mutation

# PSO-GRAANK Configurations:
VELOCITY = 0.9  # higher values helps to move to next number in search space
PERSONAL_COEFF = 0.01
GLOBAL_COEFF = 0.9
TARGET = 1
TARGET_ERROR = 1e-6
N_PARTICLES = 5

# PLS-GRAANK Configurations
STEP_SIZE = 0.5

# CluGRAD Configurations
ERASURE_PROBABILITY = 0.5  # determines the number of pairs to be ignored
SCORE_VECTOR_ITERATIONS = 10  # maximum iteration for score vector estimation

# -------- DATA SET PREPARATION -------------


class DataGP:
    """Description of class DataGP

    A class for creating data-gp objects. A data-gp object is meant to store all the parameters required by GP
    algorithms to extract gradual patterns (GP). It takes a numeric file (in CSV format) as input and converts it into
    an object whose attributes are used by algorithms to extract GPs.

        A GP is a set of gradual items (GI) and its quality is measured by its computed support value. For example given
    a data set with 3 columns (age, salary, cars) and 10 objects. A GP may take the form: {age+, salary-} with a support
    of 0.8. This implies that 8 out of 10 objects have the values of column age 'increasing' and column 'salary'
    decreasing.

    The class provides the following attributes:
        thd_supp: minimum support threshold

        equal: eq value

        titles: column names of data source

        data: all the objects organized into their respective column

        row_count: number of objects

        col_count: number of all columns

        time_cols: column indices of the columns with data-time objects

        attr_cols: column indices of the columns with numeric values

        valid_bins: valid bitmaps (in the form of ndarray) of all gradual items corresponding to the attr_cols,
        a bitmap is valid if its computed support is equal or greater than the minimum support threshold

        no_bins: True if all none of the attr_cols yields a valid bitmap

        gradual_patterns: list of GP objects


    The class provides the following functions:
        get_attr_cols: retrieves all the columns with data that is numeric and not date-tme

        get_time_cols: retrieves the columns with date-time values

        get_gi_bitmap: computes and returns the bitmap matrix corresponding to a GI

        init_attributes: generates all the bitmap matrices of valid GIs

        read (static): reads contents of a CSV file or data-frame

        test_time (static): tests if a str represents a date-time variable

        clean_data (static): cleans data (missing values, outliers) before extraction of GPs.

    """

    def __init__(self, data_source, min_sup=MIN_SUPPORT, eq=False):
        """Description of class DataGP


        A class for creating data-gp objects. A data-gp object is meant to store all the parameters required by GP
        algorithms to extract gradual patterns (GP). It takes a numeric file (in CSV format) as input and converts it
        into an object whose attributes are used by algorithms to extract GPs.

        It provides the following attributes:
            thd_supp: minimum support threshold

            equal: eq value

            titles: column names of data source

            data: all the objects organized into their respective column

            row_count: number of objects

            col_count: number of all columns

            time_cols: column indices of the columns with data-time objects

            attr_cols: column indices of the columns with numeric values

            valid_bins: valid bitmaps (in the form of ndarray) of all gradual items corresponding to the attr_cols,
            a bitmap is valid if its computed support is equal or greater than the minimum support threshold

            net_wins: a net-wins matrix constructed from valid gradual item bitmaps

            no_bins: True if all none of the attr_cols yields a valid bitmap

            gradual_patterns: list of GP objects

        :param data_source: [required] a data source, it can either be a 'file in csv format' or a 'Pandas DataFrame'

        :type data_source: str

        :param min_sup: [optional] minimum support threshold, the default is 0.5

        :param eq: encode equal values as gradual, the default is False

        """
        self.thd_supp = min_sup
        """:type thd_supp: float"""
        self.equal = eq
        """:type eq: bool"""
        self.titles, self.data = DataGP.read(data_source)
        """:type titles: ndarray"""
        """:type data: ndarray"""
        self.row_count, self.col_count = self.data.shape
        self.time_cols = self._get_time_cols()
        self.attr_cols = self._get_attr_cols()
        self.valid_bins = np.array([])
        self.no_bins = False
        self.step_name = ''  # For T-GRAANK
        self.attr_size = 0  # For T-GRAANK
        self.gradual_patterns = None

    def _get_attr_cols(self):
        """Description

        Returns indices of all columns with non-datetime objects

        :return: ndarray
        """
        all_cols = np.arange(self.col_count)
        attr_cols = np.setdiff1d(all_cols, self.time_cols)
        return attr_cols

    def _get_time_cols(self):
        """Description

        Tests each column's objects for date-time values. Returns indices of all columns with date-time objects

        :return: ndarray
        """
        # Retrieve first column only
        time_cols = list()
        n = self.col_count
        for i in range(n):  # check every column/attribute for time format
            row_data = str(self.data[0][i])
            try:
                time_ok, t_stamp = DataGP.test_time(row_data)
                if time_ok:
                    time_cols.append(i)
            except ValueError:
                continue
        return np.array(time_cols)

    def get_gi_bitmap(self, col):
        """Description

        Computes and returns the bitmap matrix corresponding to an attribute.

        :param col: specific attribute (or column)
        :return: numpy (bitmap)
        """
        if col in self.time_cols:
            raise Exception("Error: " + str(self.titles[col][1].decode()) + " is a date/time column!")
        elif col >= self.col_count:
            raise Exception("Error: Column does not exist!")
        else:
            attr_data = self.data.T
            # n = d_set.row_count
            col_data = np.array(attr_data[col], dtype=float)
            with np.errstate(invalid='ignore'):
                temp_pos = np.where(col_data < col_data[:, np.newaxis], 1, 0)
            return temp_pos

    def fit_bitmap(self, attr_data=None):
        """Description

        Generates bitmaps for columns with numeric objects. It only stores (attribute valid_bins) those bitmaps whose
        computed support values are greater or equal to the minimum support threshold value.

        :param attr_data: stepped attribute objects
        :return: void
        """
        # (check) implement parallel multiprocessing
        # 1. Transpose csv array data
        if attr_data is None:
            attr_data = self.data.T
            self.attr_size = self.row_count
        else:
            self.attr_size = len(attr_data[self.attr_cols[0]])

        # 2. Construct and store 1-item_set valid bins
        # execute binary rank to calculate support of pattern
        n = self.attr_size
        valid_bins = list()
        for col in self.attr_cols:
            col_data = np.array(attr_data[col], dtype=float)
            incr = np.array((col, '+'), dtype='i, S1')
            decr = np.array((col, '-'), dtype='i, S1')

            # 2a. Generate 1-itemset gradual items
            with np.errstate(invalid='ignore'):
                if not self.equal:
                    temp_pos = col_data < col_data[:, np.newaxis]
                else:
                    temp_pos = col_data <= col_data[:, np.newaxis]
                    np.fill_diagonal(temp_pos, 0)

                # 2b. Check support of each generated itemset
                supp = float(np.sum(temp_pos)) / float(n * (n - 1.0) / 2.0)
                if supp >= self.thd_supp:
                    valid_bins.append(np.array([incr.tolist(), temp_pos], dtype=object))
                    valid_bins.append(np.array([decr.tolist(), temp_pos.T], dtype=object))
        self.valid_bins = np.array(valid_bins)
        # print(self.valid_bins)
        if len(self.valid_bins) < 3:
            self.no_bins = True
        gc.collect()

    @staticmethod
    def read(data_src):
        """Description

        Reads all the contents of a file (in CSV format) or a data-frame. Checks if its columns have numeric values. It
        separates its columns headers (titles) from the objects.

        :param data_src:
        :return: title, column objects
        """
        # 1. Retrieve data set from source
        if isinstance(data_src, pd.DataFrame):
            # a. DataFrame source
            # Check column names
            try:
                # Check data type
                _ = data_src.columns.astype(float)

                # Add column values
                data_src.loc[-1] = data_src.columns.to_numpy(dtype=float)  # adding a row
                data_src.index = data_src.index + 1  # shifting index
                data_src.sort_index(inplace=True)

                # Rename column names
                vals = ['col_' + str(k) for k in np.arange(data_src.shape[1])]
                data_src.columns = vals
            except ValueError:
                pass
            except TypeError:
                pass
            print("Data fetched from DataFrame")
            return DataGP.clean_data(data_src)
        else:
            # b. CSV file
            file = str(data_src)
            try:
                with open(file, 'r') as f:
                    dialect = csv.Sniffer().sniff(f.readline(), delimiters=";,' '\t")
                    f.seek(0)
                    reader = csv.reader(f, dialect)
                    raw_data = list(reader)
                    f.close()

                if len(raw_data) <= 1:
                    raise Exception("CSV file read error. File has little or no data")
                else:
                    # print("Data fetched from CSV file")
                    # 2. Get table headers
                    keys = np.arange(len(raw_data[0]))
                    if raw_data[0][0].replace('.', '', 1).isdigit() or raw_data[0][0].isdigit():
                        vals = ['col_' + str(k) for k in keys]
                        header = np.array(vals, dtype='S')
                    else:
                        if raw_data[0][1].replace('.', '', 1).isdigit() or raw_data[0][1].isdigit():
                            vals = ['col_' + str(k) for k in keys]
                            header = np.array(vals, dtype='S')
                        else:
                            header = np.array(raw_data[0], dtype='S')
                            raw_data = np.delete(raw_data, 0, 0)
                    d_frame = pd.DataFrame(raw_data, columns=header)
                    return DataGP.clean_data(d_frame)
            except Exception as error:
                raise Exception("Error: " + str(error))

    @staticmethod
    def test_time(date_str):
        """Description

        Tests if a str represents a date-time variable.

        :param date_str: str value
        :return: bool (True if it is date-time variable, False otherwise)
        """
        # add all the possible formats
        try:
            if type(int(date_str)):
                return False, False
        except ValueError:
            try:
                if type(float(date_str)):
                    return False, False
            except ValueError:
                try:
                    date_time = parse(date_str)
                    t_stamp = time.mktime(date_time.timetuple())
                    return True, t_stamp
                except ValueError:
                    raise ValueError('no valid date-time format found')

    @staticmethod
    def clean_data(df):
        """Description

        Cleans a data-frame (i.e., missing values, outliers) before extraction of GPs
        :param df: data-frame
        :return: list (column titles), numpy (cleaned data)
        """
        # 1. Remove objects with Null values
        df = df.dropna()

        # 2. Remove columns with Strings
        cols_to_remove = []
        for col in df.columns:
            try:
                _ = df[col].astype(float)
            except ValueError:
                # Keep time columns
                try:
                    ok, stamp = DataGP.test_time(str(df[col][0]))
                    if not ok:
                        cols_to_remove.append(col)
                except ValueError:
                    cols_to_remove.append(col)
                pass
            except TypeError:
                cols_to_remove.append(col)
                pass
        # keep only the columns in df that do not contain string
        df = df[[col for col in df.columns if col not in cols_to_remove]]

        # 3. Return titles and data
        if df.empty:
            raise Exception("Data set is empty after cleaning.")

        keys = np.arange(df.shape[1])
        values = np.array(df.columns, dtype='S')
        titles = np.rec.fromarrays((keys, values), names=('key', 'value'))
        # print("Data cleaned")
        return titles, df.values


class DfsDataGP:

    def __init__(self, file_path, min_sup=0):
        self.thd_supp = min_sup
        self.titles, self.data = DataGP.read(file_path)
        self.row_count, self.col_count = self.data.shape
        self.time_cols = self.get_time_cols()
        self.attr_cols = self.get_attr_cols()
        self.cost_matrix = np.ones((self.col_count, 3), dtype=int)
        self.min_len = self.thd_supp * self.row_count
        self.item_to_tids = defaultdict(set)
        self.gradual_patterns = None
        # self.encoded_data = np.array([])

    def get_attr_cols(self):
        """
        Returns indices of all columns with non-datetime objects

        :return: ndarray
        """
        all_cols = np.arange(self.col_count)
        attr_cols = np.setdiff1d(all_cols, self.time_cols)
        return attr_cols

    def get_time_cols(self):
        """
        Tests each column's objects for date-time values. Returns indices of all columns with date-time objects

        :return: ndarray
        """
        # Retrieve first column only
        time_cols = list()
        n = self.col_count
        for i in range(n):  # check every column/attribute for time format
            row_data = str(self.data[0][i])
            try:
                time_ok, t_stamp = DataGP.test_time(row_data)
                if time_ok:
                    time_cols.append(i)
            except ValueError:
                continue
        return np.array(time_cols)

    def encode_data(self):
        attr_data = self.data.T
        # self.attr_size = len(attr_data[self.attr_cols[0]])
        size = self.row_count  # self.attr_size
        n = len(self.attr_cols) + 2
        encoded_data = list()
        for i in range(size):
            j = i + 1
            if j >= size:
                continue

            temp_arr = np.empty([n, (size - j)], dtype=int)
            temp_arr[0] = np.repeat(i, (size - j))
            temp_arr[1] = np.arange(j, size)
            k = 2
            for col in self.attr_cols:
                row_in = attr_data[col][i]
                row_js = attr_data[col][(i+1):size]
                v = col + 1
                row = np.where(row_js > row_in, v, np.where(row_js < row_in, -v, 0))
                temp_arr[k] = row
                k += 1
                pos_cost = np.count_nonzero(row == v)
                neg_cost = np.count_nonzero(row == -v)
                inv_cost = np.count_nonzero(row == 0)
                self.cost_matrix[col][0] += (neg_cost + inv_cost)
                self.cost_matrix[col][1] += (pos_cost + inv_cost)
                self.cost_matrix[col][2] += (pos_cost + neg_cost)
            temp_arr = temp_arr.T
            encoded_data.extend(temp_arr)
        self.data = None
        gc.collect()
        return np.array(encoded_data)

    def remove_inv_attrs(self, encoded_data):
        c_matrix = self.cost_matrix
        # 1. remove invalid attributes
        valid_a1 = list()
        valid_a2 = [-2, -1]
        for i in range(len(self.attr_cols)):
            a = self.attr_cols[i]
            valid = (c_matrix[a][0] < c_matrix[a][2]) or \
                    (c_matrix[a][1] < c_matrix[a][2])
            if valid:
                valid_a1.append(i)
                valid_a2.append(i)
        self.attr_cols = self.attr_cols[valid_a1]
        valid_a2 = np.array(valid_a2) + 2
        encoded_data = encoded_data[:, valid_a2]
        return encoded_data

    def init_transaction_ids(self):

        encoded_data = self.remove_inv_attrs(self.encode_data())

        # 1. group similar items
        for t in range(len(encoded_data)):
            transaction = encoded_data[t][2:]
            for item in transaction:
                self.item_to_tids[item].add(tuple(encoded_data[t][:2]))

        low_supp_items = [k for k, v in self.item_to_tids.items()
                          if len(np.unique(np.array(list(v))[:, 0], axis=0))
                          < self.min_len]
        for item in low_supp_items:
            del self.item_to_tids[item]
        gc.collect()


# -------- OTHER METHODS -----------

def analyze_gps(file, min_sup, est_gps, approach='bfs'):
    """Description

    For each estimated GP, computes its true support using GRAANK approach and returns the statistics (% error,
    and standard deviation).

    :param file: data set file
    :param min_sup: minimum support (set by user)
    :param est_gps: estimated GPs
    :param approach: 'bfs' (default) or 'dfs'
    :return: tabulated results
    """
    if approach == 'dfs':
        d_set = DfsDataGP(file, min_sup)
        d_set.init_transaction_ids()
    else:
        d_set = DataGP(file, min_sup)
        d_set.fit_bitmap()
    headers = ["Gradual Pattern", "Estimated Support", "True Support", "Percentage Error", "Standard Deviation"]
    data = []
    for est_gp in est_gps:
        est_sup = est_gp.support
        est_gp.set_support(0)
        if approach == 'dfs':
            true_gp = est_gp.validate_tree(d_set)
        else:
            true_gp = est_gp.validate_graank(d_set)
        true_sup = true_gp.support

        if true_sup == 0:
            percentage_error = np.inf
            st_dev = np.inf
        else:
            percentage_error = ((est_sup - true_sup) / true_sup) * 100
            st_dev = statistics.stdev([est_sup, true_sup])

        if len(true_gp.gradual_items) == len(est_gp.gradual_items):
            data.append([est_gp.to_string(), round(est_sup, 3), round(true_sup, 3), str(round(percentage_error, 3))+'%',
                         round(st_dev, 3)])
        else:
            data.append([est_gp.to_string(), round(est_sup, 3), -1, np.inf, np.inf])
    return tabulate(data, headers=headers)


def get_num_cores():
    """Description

    Finds the count of CPU cores in a computer or a SLURM super-computer.
    :return: number of cpu cores (int)
    """
    num_cores = get_slurm_cores()
    if not num_cores:
        num_cores = mp.cpu_count()
    return num_cores


def get_slurm_cores():
    """Description

    Test computer to see if it is a SLURM environment, then gets number of CPU cores.
    :return: count of CPUs (int) or False
    """
    try:
        cores = int(os.environ['SLURM_JOB_CPUS_PER_NODE'])
        return cores
    except ValueError:
        try:
            str_cores = str(os.environ['SLURM_JOB_CPUS_PER_NODE'])
            temp = str_cores.split('(', 1)
            cpus = int(temp[0])
            str_nodes = temp[1]
            temp = str_nodes.split('x', 1)
            str_temp = str(temp[1]).split(')', 1)
            nodes = int(str_temp[0])
            cores = cpus * nodes
            return cores
        except ValueError:
            return False
    except KeyError:
        return False


def write_file(data, path, wr=True):
    """Description

    Writes data into a file
    :param data: information to be written
    :param path: name of file and storage path
    :param wr: writes data into file if True
    :return:
    """
    if wr:
        with open(path, 'w') as f:
            f.write(data)
            f.close()
    else:
        pass


# -------- GRADUAL PATTERNS -------------


class GI:
    """Description of class GI (Gradual Item)

    A class that is used to create GI objects. A GI is a pair (i,v) where i is a column and v is a variation symbol -
    increasing/decreasing. Each column of a data set yields 2 GIs; for example, column age yields GI age+ or age-.

    An example representation of a GI object: (0, +)

    This class has the following attributes:
        attribute_col: column index of a data set

        symbol: variation symbol (either + or -)

        gradual_item: the GI in a ndarray format

        tuple: the GI in a tuple format

        rank_sum:

    """

    def __init__(self, attr_col, symbol):
        """Description of class GI (Gradual Item)

        A class that is used to create GI objects. A GI is a pair (i,v) where is a column and v is a variation symbol -
        increasing/decreasing. Each column of a data set yields 2 GIs; for example, column age yields GI age+ or age-.

        This class has the following attributes:
            attribute_col: column index of a data set

            symbol: variation symbol (either + or -)

            gradual_item: the GI in a ndarray format

            tuple: the GI in a tuple format

            rank_sum:

        :param attr_col: column index
        :param symbol: variation symbol (either + or -)
        """
        self.attribute_col = attr_col
        """:type attribute_col: int"""
        self.symbol = symbol
        """:type symbol: str"""
        self.gradual_item = np.array((attr_col, symbol), dtype='i, S1')
        self.tuple = tuple([attr_col, symbol])
        self.rank_sum = 0

    def inv(self):
        """Description

        Inverts a GI to the opposite variation (i.e., from - to +; or, from + to -)
        :return: inverted GI (ndarray)
        """
        if self.symbol == '+':
            # temp = tuple([self.attribute_col, '-'])
            temp = np.array((self.attribute_col, '-'), dtype='i, S1')
        elif self.symbol == '-':
            # temp = tuple([self.attribute_col, '+'])
            temp = np.array((self.attribute_col, '+'), dtype='i, S1')
        else:
            temp = np.array((self.attribute_col, 'x'), dtype='i, S1')
        return temp

    def inv_gi(self):
        """Description

        Inverts a GI to the opposite variation (i.e., from - to +; or, from + to -)
        :return: inverted GI object
        """
        if self.symbol == '+':
            sym = '-'
        else:
            sym = '+'
        new_gi = GI(self.attribute_col, sym)
        return new_gi

    def as_integer(self):
        """Description

        Converts variation symbol into an integer (i.e., + to 1; and - to -1)
        :return: GI with an integer variation symbol
        """
        if self.symbol == '+':
            temp = [self.attribute_col, 1]
        elif self.symbol == '-':
            temp = [self.attribute_col, -1]
        else:
            temp = [self.attribute_col, 0]
        return temp

    def as_string(self):
        """Description

        Stringifies a GI. It converts variation symbol into a string (i.e., + to _pos; and - to _neg)
        :return: GI with a string variation symbol
        """
        if self.symbol == '+':
            temp = str(self.attribute_col) + '_pos'
        elif self.symbol == '-':
            temp = str(self.attribute_col) + '_neg'
        else:
            temp = str(self.attribute_col) + '_inv'
        return temp

    def to_string(self):
        """Description

        Returns a GI in string format
        :return: string
        """
        return str(self.attribute_col) + self.symbol

    def is_decrement(self):
        """Description

        Checks if a GI's variation corresponds to decreasing
        :return: True is GI has a decreasing variation, False otherwise
        """
        if self.symbol == '-':
            return True
        else:
            return False

    @staticmethod
    def parse_gi(gi_str):
        """Description

        Converts a stringified GI into normal GI.
        :param gi_str: stringified GI
        :return: GI
        """
        txt = gi_str.split('_')
        attr_col = int(txt[0])
        if txt[1] == 'neg':
            symbol = '-'
        else:
            symbol = '+'
        return GI(attr_col, symbol)

    @staticmethod
    def inv_arr(g_item):
        """Description

        Computes the inverse of a GI formated as an array or tuple

        :param g_item: gradual item (array/tuple)
        :return: inverted gradual item
        """
        if g_item[1] == '+':
            temp = tuple([g_item[0], '-'])
        else:
            temp = tuple([g_item[0], '+'])
        return temp


class GP:
    """Description of class GP (Gradual Pattern)

    A class that is used to create GP objects. a GP object is a set of gradual items (GI) and its quality is measured by
    its computed support value. For example given a data set with 3 columns (age, salary, cars) and 10 objects. A GP may
    take the form: {age+, salary-} with a support of 0.8. This implies that 8 out of 10 objects have the values of
    column age 'increasing' and column 'salary' decreasing.

    An example representation of a GP object: {(0, +), (1, -), (3, +)}

     The class has the following attributes:
        gradual_items: list if GIs

        support: computed support value as a float

    """

    def __init__(self):
        """Description of class GP (Gradual Pattern)

            A class that is used to create GP objects. a GP object is a set of gradual items (GI) and its quality is
            measured by its computed support value. For example given a data set with 3 columns (age, salary,
            cars) and 10 objects. A GP may take the form: {age+, salary-} with a support of 0.8. This implies that 8
            out of 10 objects have the values of column age 'increasing' and column 'salary' decreasing.

             The class has the following attributes:
                gradual_items: list if GIs

                support: computed support value as a float

            """
        self.gradual_items = list()
        self.support = 0
        """:type support: float"""

    def set_support(self, support):
        """Description

        Sets the computed support value of the gradual pattern (GP)
        :param support: support value
        :return: void
        """
        self.support = round(support, 3)

    def add_gradual_item(self, item):
        """Description

        Adds a gradual item (GI) into the gradual pattern (GP)
        :param item: gradual item
        :return: void
        """
        if item.symbol == '-' or item.symbol == '+':
            self.gradual_items.append(item)
        else:
            pass

    def add_items_from_list(self, lst_items):
        """Description

        Adds gradual items from a list of str or a list of sets.

        :param lst_items: str or set
        :return: none
        """
        for str_gi in lst_items:
            if type(str_gi[1]) is str:
                self.add_gradual_item(GI(int(str_gi[0]), str_gi[1]))
            elif type(str_gi[1]) is bytes:
                self.add_gradual_item(GI(int(str_gi[0]), str(str_gi[1].decode())))

    def get_pattern(self):
        """Description

        Returns the gradual pattern (GP) as a list
        :return: gradual pattern
        """
        pattern = list()
        for item in self.gradual_items:
            pattern.append(item.gradual_item.tolist())
        return pattern

    def get_np_pattern(self):
        """Description

        Returns a gradual pattern (GP) as a ndarray
        :return: ndarray
        """
        pattern = []
        for item in self.gradual_items:
            pattern.append(item.gradual_item)
        return np.array(pattern)

    def get_tuples(self):
        """Description

        Returns the gradual pattern (GP) as a list of GI tuples
        :return: list of GI tuples
        """
        pattern = list()
        for gi in self.gradual_items:
            temp = tuple([gi.attribute_col, gi.symbol])
            pattern.append(temp)
        return pattern

    def get_attributes(self):
        """Description

        Breaks down all the gradual items (GIs) in the gradual pattern into columns and variation symbols and returns
        them as separate variables
        :return: separate columns and variation symbols
        """
        attrs = list()
        syms = list()
        for item in self.gradual_items:
            gi = item.as_integer()
            attrs.append(gi[0])
            syms.append(gi[1])
        return attrs, syms

    def get_index(self, gi):
        """Description

        Returns the index position of a gradual item in the gradual pattern
        :param gi: gradual item
        :return: index of gradual item
        """
        for i in range(len(self.gradual_items)):
            gi_obj = self.gradual_items[i]
            if (gi.symbol == gi_obj.symbol) and (gi.attribute_col == gi_obj.attribute_col):
                return i
        return -1

    def inv_pattern(self):
        """Description

        Inverts all the variation symbols of all the gradual items (GIs) in the gradual pattern (GP)
        :return: inverted GP
        """
        pattern = list()
        for gi in self.gradual_items:
            pattern.append(gi.inv().tolist())
        return pattern

    def contains(self, gi):
        """Description

        Checks if a gradual item (GI) is a member of a gradual pattern (GP)
        :param gi: gradual item
        :return: True if it is a member, otherwise False
        """
        if gi is None:
            return False
        if gi in self.gradual_items:
            return True
        return False

    def contains_strict(self, gi):
        """Description

        Strictly checks if a gradual item (GI) is a member of a gradual pattern (GP)
        :param gi: gradual item
        :return: True if it is a member, otherwise False
        """
        if gi is None:
            return False
        for gi_obj in self.gradual_items:
            if (gi.attribute_col == gi_obj.attribute_col) and (gi.symbol == gi_obj.symbol):
                return True
        return False

    def contains_attr(self, gi):
        """Description

        Checks is any gradual item (GI) in the gradual pattern (GP) is composed of the column
        :param gi: gradual item
        :return: True if column exists, False otherwise
        """
        if gi is None:
            return False
        for gi_obj in self.gradual_items:
            if gi.attribute_col == gi_obj.attribute_col:
                return True
        return False

    def to_string(self):
        """Description

        Returns the GP in string format
        :return: string
        """
        pattern = list()
        for item in self.gradual_items:
            pattern.append(item.to_string())
        return pattern

    def to_dict(self):
        """Description

        Returns the GP as a dictionary
        :return: dict
        """
        gi_dict = {}
        for gi in self.gradual_items:
            gi_dict.update({gi.as_string(): 0})
        return gi_dict

    def print(self, columns):
        """Description

        Returns patterns with actual column names
        :param columns: Columns names
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
    """Description of class ExtGP (Extended Gradual Pattern)

    A class that inherits class GP which is used to create more capable GP objects. a GP object is a set of gradual
    items and its quality is measured by its computed support value. For example given a data set with 3 columns
    (age, salary, cars) and 10 objects. A GP may take the form: {age+, salary-} with a support of 0.8. This implies that
    8 out of 10 objects have the values of column age 'increasing' and column 'salary' decreasing.

    The class GP has the following attributes:
        gradual_items: list if GIs

        support: computed support value as a float

    The class ExtGP adds the following functions:
        validate: used to validate GPs

        check_am: used to verify if a GP obeys anti-monotonicity

        is_duplicate: checks a GP is already extracted

    """

    def __init__(self):
        """Description of class ExtGP (Extended Gradual Pattern)

        A class that inherits class GP which is used to create more powerful GP objects that can be used in mining
        approaches that implement swarm optimization techniques or cluster analysis or classification algorithms.

        It adds the following attribute:
            freq_count: frequency count of a particular GP object.

        """
        super(ExtGP, self).__init__()
        self.freq_count = 0
        """:type freq_count: int"""

    def validate_graank(self, d_set):
        """Description

        Validates a candidate gradual pattern (GP) based on support computation. A GP is invalid if its support value is
        less than the minimum support threshold set by the user.

        :param d_set: Data_GP object
        :return: a valid GP or an empty GP
        """
        # pattern = [('2', '+'), ('4', '+')]
        min_supp = d_set.thd_supp
        n = d_set.attr_size
        gen_pattern = ExtGP()
        """type gen_pattern: ExtGP"""
        bin_arr = np.array([])

        for gi in self.gradual_items:
            arg = np.argwhere(np.isin(d_set.valid_bins[:, 0], gi.gradual_item))
            if len(arg) > 0:
                i = arg[0][0]
                valid_bin = d_set.valid_bins[i]
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

    def validate_tree(self, d_set):
        min_supp = d_set.thd_supp
        n = d_set.row_count
        gen_pattern = ExtGP()
        """type gen_pattern: ExtGP"""
        temp_tids = None
        for gi in self.gradual_items:
            gi_int = gi.as_integer()
            node = int(gi_int[0] + 1) * gi_int[1]
            gi_int = (gi.inv_gi()).as_integer()
            node_inv = int(gi_int[0] + 1) * gi_int[1]
            for k, v in d_set.item_to_tids.items():
                if node == k:
                    if temp_tids is None:
                        temp_tids = v
                        gen_pattern.add_gradual_item(gi)
                    else:
                        temp = temp_tids.copy()
                        temp = temp.intersection(v)
                        if len(temp) > 1:
                            x = np.unique(np.array(list(temp))[:, 0], axis=0)
                            supp = len(x) / n
                        else:
                            supp = len(temp) / n

                        if supp >= min_supp:
                            temp_tids = temp.copy()
                            gen_pattern.add_gradual_item(gi)
                            gen_pattern.set_support(supp)
                elif node_inv == k:
                    if temp_tids is None:
                        temp_tids = v
                        gen_pattern.add_gradual_item(gi)
                    else:
                        temp = temp_tids.copy()
                        temp = temp.intersection(v)
                        if len(temp) > 1:
                            x = np.unique(np.array(list(temp))[:, 0], axis=0)
                            supp = len(x) / n
                        else:
                            supp = len(temp) / n

                        if supp >= min_supp:
                            temp_tids = temp.copy()
                            gen_pattern.add_gradual_item(gi)
                            gen_pattern.set_support(supp)
        if len(gen_pattern.gradual_items) <= 1:
            return self
        else:
            return gen_pattern

    def check_am(self, gp_list, subset=True):
        """Description

        Anti-monotonicity check. Checks if a GP is a subset or superset of an already existing GP

        :param gp_list: list of existing GPs
        :param subset: check if it is a subset
        :return: True if superset/subset, False otherwise
        """
        result = False
        if subset:
            for pat in gp_list:
                result1 = set(self.get_pattern()).issubset(set(pat.get_pattern()))
                result2 = set(self.inv_pattern()).issubset(set(pat.get_pattern()))
                if result1 or result2:
                    result = True
                    break
        else:
            for pat in gp_list:
                result1 = set(self.get_pattern()).issuperset(set(pat.get_pattern()))
                result2 = set(self.inv_pattern()).issuperset(set(pat.get_pattern()))
                if result1 or result2:
                    result = True
                    break
        return result

    def is_duplicate(self, valid_gps, invalid_gps=None):
        """Description

        Checks if a pattern is in the list of winner GPs or loser GPs

        :param valid_gps: list of GPs
        :param invalid_gps: list of GPs
        :return: True if pattern is either list, False otherwise
        """
        if invalid_gps is None:
            pass
        else:
            for pat in invalid_gps:
                if set(self.get_pattern()) == set(pat.get_pattern()) or \
                        set(self.inv_pattern()) == set(pat.get_pattern()):
                    return True
        for pat in valid_gps:
            if set(self.get_pattern()) == set(pat.get_pattern()) or \
                    set(self.inv_pattern()) == set(pat.get_pattern()):
                return True
        return False


class NumericSS:
    """Description of class NumericSS (Numeric Search Space)

    A class that implements functions that allow swarm algorithms to explore a numeric search space.

    The class NumericSS has the following functions:
        decode_gp: decodes a GP from a numeric position
        cost_function: computes the fitness of a GP
        apply_bound: applies minimum and maximum values

    """

    @staticmethod
    def decode_gp(attr_keys, position):
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
    def cost_function(position, attr_keys, d_set):
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
            arg = np.argwhere(np.isin(d_set.valid_bins[:, 0], gi.gradual_item))
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
    def apply_bound(x, var_min, var_max):
        """Description

        Modifies x (a numeric value) if it exceeds the lower/upper bound of the numeric search space.

        :param x: a value in the numeric search space
        :param var_min: lower-bound value
        :param var_max: upper-bound value
        :return: nothing
        """

        x.position = np.maximum(x.position, var_min)
        x.position = np.minimum(x.position, var_max)


class TimeLag:

    def __init__(self, tstamp=0, supp=0):
        self.timestamp = tstamp
        self.support = round(supp, 3)
        self.sign = self.get_sign()
        if tstamp == 0:
            self.time_lag = np.array([])
            self.valid = False
        else:
            self.time_lag = np.array(self.format_time())
            self.valid = True

    def get_sign(self):
        if self.timestamp < 0:
            sign = "-"
        else:
            sign = "+"
        return sign

    def format_time(self):
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

    def to_string(self):
        if len(self.time_lag) > 0:
            txt = ("~ " + self.sign + str(self.time_lag[0]) + " " + str(self.time_lag[1])
                   + " : " + str(self.support))
        else:
            txt = "No time lag found!"
        return txt


# --------- ALGORITHMS ---------------------


class AntGRAANK(DataGP):
    """Description of class AntGRAANK

    Extract gradual patterns (GPs) from a numeric data source using the Ant Colony Optimization approach
    (proposed in a published paper by Dickson Owuor). A GP is a set of gradual items (GI) and its quality is
    measured by its computed support value. For example given a data set with 3 columns (age, salary, cars) and 10
    objects. A GP may take the form: {age+, salary-} with a support of 0.8. This implies that 8 out of 10 objects
    have the values of column age 'increasing' and column 'salary' decreasing.

        In this approach, it is assumed that every column can be converted into gradual item (GI). If the GI is valid
        (i.e. its computed support is greater than the minimum support threshold) then it is either increasing or
        decreasing (+ or -), otherwise it is irrelevant (x). Therefore, a pheromone matrix is built using the number of
        columns and the possible variations (increasing, decreasing, irrelevant) or (+, -, x). The algorithm starts by
        randomly generating GP candidates using the pheromone matrix, each candidate is validated by confirming that
        its computed support is greater or equal to the minimum support threshold. The valid GPs are used to update the
        pheromone levels and better candidates are generated.

    This class extends class DataGP, and it provides the following additional attributes:

        max_iteration: integer value determines the number of iterations for the algorithm

        evaporation_factor: value between 0-1 which determines how fast pheromone levels evaporate

        distance_matrix: an array that stores the cost between travelling between nodes

        attribute_keys: an array with attribute keys

    """

    def __init__(self, *args, max_iter=MAX_ITERATIONS, e_factor=EVAPORATION_FACTOR):
        """Description

    Extract gradual patterns (GPs) from a numeric data source using the Ant Colony Optimization approach
    (proposed in a published paper by Dickson Owuor). A GP is a set of gradual items (GI) and its quality is
    measured by its computed support value. For example given a data set with 3 columns (age, salary, cars) and 10
    objects. A GP may take the form: {age+, salary-} with a support of 0.8. This implies that 8 out of 10 objects
    have the values of column age 'increasing' and column 'salary' decreasing.

        In this approach, it is assumed that every column can be converted into gradual item (GI). If the GI is valid
        (i.e. its computed support is greater than the minimum support threshold) then it is either increasing or
        decreasing (+ or -), otherwise it is irrelevant (x). Therefore, a pheromone matrix is built using the number of
        columns and the possible variations (increasing, decreasing, irrelevant) or (+, -, x). The algorithm starts by
        randomly generating GP candidates using the pheromone matrix, each candidate is validated by confirming that
        its computed support is greater or equal to the minimum support threshold. The valid GPs are used to update the
        pheromone levels and better candidates are generated.

    This class extends class DataGP, and it provides the following additional attributes:

        max_iteration: integer value determines the number of iterations for the algorithm

        evaporation_factor: value between 0-1 which determines how fast pheromone levels evaporate

        distance_matrix: an array that stores the cost between travelling between nodes

        attribute_keys: an array with attribute keys

        :param args: [required] data-source, [optional] minimum-support
        :param max_iter: maximum_iteration, default is 1
        :param e_factor: evaporation factor, default is 0.5
        """
        super(AntGRAANK, self).__init__(*args)
        self.evaporation_factor = e_factor
        self.max_iteration = max_iter
        self.distance_matrix = None
        self.attribute_keys = None

    def _fit(self):
        """Description

        Generates the distance matrix (d)
        :return: distance matrix (d) and attribute keys
        """
        v_bins = self.valid_bins
        # 1. Fetch valid bins group
        attr_keys = [GI(x[0], x[1].decode()).as_string() for x in v_bins[:, 0]]

        # 2. Initialize an empty d-matrix
        n = len(attr_keys)
        d = np.zeros((n, n), dtype=np.dtype('i8'))  # cumulative sum of all segments
        for i in range(n):
            for j in range(n):
                if GI.parse_gi(attr_keys[i]).attribute_col == GI.parse_gi(attr_keys[j]).attribute_col:
                    # Ignore similar attributes (+ or/and -)
                    continue
                else:
                    bin_1 = v_bins[i][1]
                    bin_2 = v_bins[j][1]
                    # Cumulative sum of all segments for 2x2 (all attributes) gradual items
                    d[i][j] += np.sum(np.multiply(bin_1, bin_2))
        # print(d)
        self.distance_matrix = d
        self.attribute_keys = attr_keys
        gc.collect()

    def _gen_aco_candidates(self, p_matrix):
        """Description

        Generates GP candidates based on the pheromone levels.

        :param p_matrix: pheromone matrix (ndarray)
        :return: pheromone matrix (ndarray)
        """
        v_matrix = self.distance_matrix
        pattern = ExtGP()
        ":type pattern: ExtGP"

        # 1. Generate gradual items with the highest pheromone and visibility
        m = p_matrix.shape[0]
        for i in range(m):
            combine_feature = np.multiply(v_matrix[i], p_matrix[i])
            total = np.sum(combine_feature)
            with np.errstate(divide='ignore', invalid='ignore'):
                probability = combine_feature / total
            cum_prob = np.cumsum(probability)
            r = np.random.random_sample()
            try:
                j = np.nonzero(cum_prob > r)[0][0]
                gi = GI.parse_gi(self.attribute_keys[j])
                """:type gi: GI"""
                if not pattern.contains_attr(gi):
                    pattern.add_gradual_item(gi)
            except IndexError:
                continue

        # 2. Evaporate pheromones by factor e
        p_matrix = (1 - self.evaporation_factor) * p_matrix
        return pattern, p_matrix

    def _update_pheromones(self, pattern, p_matrix):
        """Description

        Updates the pheromone level of the pheromone matrix

        :param pattern: pattern used to update values
        :param p_matrix: an existing pheromone matrix
        :return: updated pheromone matrix
        """
        idx = [self.attribute_keys.index(x.as_string()) for x in pattern.gradual_items]
        for n in range(len(idx)):
            for m in range(n + 1, len(idx)):
                i = idx[n]
                j = idx[m]
                p_matrix[i][j] += 1
                p_matrix[j][i] += 1
        return p_matrix

    def discover(self):
        """Description

        Applies ant-colony optimization algorithm and uses pheromone levels to find GP candidates. The candidates are
        validated if their computed support is greater than or equal to the minimum support threshold specified by the
        user.

        :return: JSON object
        """
        # 0. Initialize and prepare data set
        # d_set = DataGP(f_path, min_supp)
        # """:type d_set: DataGP"""
        self.fit_bitmap()
        self._fit()  # distance matrix (d) & attributes corresponding to d
        d = self.distance_matrix

        a = self.attr_size
        self.gradual_patterns = list()  # subsets
        loser_gps = list()  # supersets
        str_winner_gps = list()  # subsets
        repeated = 0
        it_count = 0
        counter = 0

        if self.no_bins:
            return []

        # 1. Remove d[i][j] < frequency-count of min_supp
        fr_count = ((self.thd_supp * a * (a - 1)) / 2)
        d[d < fr_count] = 0

        # 3. Initialize pheromones (p_matrix)
        pheromones = np.ones(d.shape, dtype=float)

        invalid_count = 0
        # 4. Iterations for ACO
        # while repeated < 1:
        while counter < self.max_iteration:
            rand_gp, pheromones = self._gen_aco_candidates(pheromones)
            if len(rand_gp.gradual_items) > 1:
                # print(rand_gp.get_pattern())
                exits = rand_gp.is_duplicate(self.gradual_patterns, loser_gps)
                if not exits:
                    repeated = 0
                    # check for anti-monotony
                    is_super = rand_gp.check_am(loser_gps, subset=False)
                    is_sub = rand_gp.check_am(self.gradual_patterns, subset=True)
                    if is_super or is_sub:
                        continue
                    gen_gp = rand_gp.validate_graank(self)
                    """:type gen_gp: ExtGP"""
                    is_present = gen_gp.is_duplicate(self.gradual_patterns, loser_gps)
                    is_sub = gen_gp.check_am(self.gradual_patterns, subset=True)
                    if is_present or is_sub:
                        repeated += 1
                    else:
                        if gen_gp.support >= self.thd_supp:
                            pheromones = self._update_pheromones(gen_gp, pheromones)
                            self.gradual_patterns.append(gen_gp)
                            str_winner_gps.append(gen_gp.print(self.titles))
                        else:
                            loser_gps.append(gen_gp)
                            invalid_count += 1
                    if set(gen_gp.get_pattern()) != set(rand_gp.get_pattern()):
                        loser_gps.append(rand_gp)
                else:
                    repeated += 1
            else:
                invalid_count += 1
            it_count += 1
            if self.max_iteration == 1:
                counter = repeated
            else:
                counter = it_count
        # Output
        out = json.dumps({"Algorithm": "ACO-GRAANK", "Best Patterns": str_winner_gps, "Invalid Count": invalid_count,
                          "Iterations": it_count})
        """:type out: object"""
        return out


class ClusterGP(DataGP):
    """Description of class CluDataGP (Clustering DataGP)

    CluDataGP stands for Clustering DataGP. It is a class that inherits the DataGP class in order to create data-gp
    objects for the clustering approach. This class inherits the DataGP class which is used to create data-gp objects.
    The classical data-gp object is meant to store all the parameters required by GP algorithms to extract gradual
    patterns (GP). It takes a numeric file (in CSV format) as input and converts it into an object whose attributes are
    used by algorithms to extract GPs.

    class DataGP provides the following attributes:
        thd_supp: minimum support threshold

        equal: eq value

        titles: column names of data source

        data: all the objects organized into their respective column

        row_count: number of objects

        col_count: number of all columns

        time_cols: column indices of the columns with data-time objects

        attr_cols: column indices of the columns with numeric values

        valid_bins: valid bitmaps (in the form of ndarray) of all gradual items corresponding to the attr_cols, a bitmap
        is valid if its computed support is equal or greater than the minimum support threshold

        no_bins: True if all none of the attr_cols yields a valid bitmap

        gradual_patterns: list of GP objects

    This class adds the parameters required for clustering gradual items to the data-gp object. The class provides the
    following additional attributes:
        e_prob: erasure probability (a value between 0 - 1)

        mat_iter: maximum iteration value for score vector estimation

    CluDataGP adds the following functions:
        construct_matrices: generates the net-win matrix

        infer_gps: infers GPs from clusters of Gradual Items

        estimate_score_vector: estimates the score vector based on the cumulative wins

        estimate_support:  estimates the frequency support of a GP based on its score vector

    """

    def __init__(self, *args, e_prob=ERASURE_PROBABILITY, max_iter=SCORE_VECTOR_ITERATIONS, no_prob=False):
        """Description of class CluDataGP (Clustering DataGP)

        A class for creating data-gp objects for the clustering approach. This class inherits the DataGP class which is
        used to create data-gp objects. This class adds the parameters required for clustering gradual items to the
        data-gp object.

        The class provides the following additional attributes:

            e_prob: erasure probability (a value between 0 - 1). Erasure probability determines the proportion of ij
            columns to be used by the algorithm, the rest are ignored.

            mat_iter: maximum iteration value for score vector estimation

        :param args: [required] data-source, [optional] minimum-support
        :param e_prob: [optional] erasure probability, the default is 0.5
        :param max_iter: [optional] maximum iteration for score vector estimation, the default is 10
        """
        super(ClusterGP, self).__init__(*args)
        self.erasure_probability = e_prob
        """:type erasure_probability: float"""
        self.max_iteration = max_iter
        """:type max_iteration: int"""
        if not no_prob:
            self.gradual_items, self.cum_wins, self.net_win_mat, self.ij = self._construct_matrices(e_prob)
            """:type gradual_items: ndarray"""
            """:type cum_wins: ndarray"""
            """:type net_win_mat: ndarray"""
            """:type ij: ndarray"""
            self.win_mat = np.array([])
            """:type win_mat: ndarray"""
        else:
            self.gradual_items, self.win_mat, self.cum_wins, self.net_win_mat, self.ij = self._construct_all_matrices()
            """:type gradual_items: ndarray"""
            """:type win_mat: ndarray"""
            """:type cum_wins: ndarray"""
            """:type net_win_mat: ndarray"""
            """:type ij: ndarray"""

    def _construct_matrices(self, e):
        """Description

        Generates all the gradual items and, constructs: (1) net-win matrix, (2) cumulative wins, (3) pairwise objects.

        :param e: [required] erasure probability
        :return: list of gradual items, net-win matrix, cumulative win matrix, selected pairwise (ij) objects
        """

        n = self.row_count
        prob = 1 - e  # Sample probability

        if prob == 1:
            # 1a. Generate all possible pairs
            pair_ij = np.array(np.meshgrid(np.arange(n), np.arange(n))).T.reshape(-1, 2)

            # 1b. Remove duplicates or reversed pairs
            pair_ij = pair_ij[np.argwhere(pair_ij[:, 0] < pair_ij[:, 1])[:, 0]]
        else:
            # 1a. Generate random pairs using erasure-probability
            total_pair_count = int(n * (n - 1) * 0.5)
            rand_1d = np.random.choice(n, int(prob * total_pair_count) * 2, replace=True)
            pair_ij = np.reshape(rand_1d, (-1, 2))

            # 1b. Remove duplicates
            pair_ij = pair_ij[np.argwhere(pair_ij[:, 0] != pair_ij[:, 1])[:, 0]]

        # 2. Variable declarations
        attr_data = self.data.T  # Feature data objects
        lst_gis = []  # List of GIs
        s_mat = []  # S-Matrix (made up of S-Vectors)
        cum_wins = []  # Cumulative wins

        # 3. Construct S matrix from data set
        for col in np.nditer(self.attr_cols):
            # Feature data objects
            col_data = np.array(attr_data[col], dtype=np.float)  # Feature data objects

            # Cumulative Wins: for estimation of score-vector
            temp_cum_wins = np.where(col_data[pair_ij[:, 0]] < col_data[pair_ij[:, 1]], 1,
                                     np.where(col_data[pair_ij[:, 0]] > col_data[pair_ij[:, 1]], -1, 0))
            # print(col)
            # print(temp_cum_wins)

            # S-vector
            s_vec = np.zeros((n,), dtype=np.int32)
            for w in [1, -1]:
                positions = np.flatnonzero(temp_cum_wins == w)
                i, counts_i = np.unique(pair_ij[positions, 0], return_counts=True)
                j, counts_j = np.unique(pair_ij[positions, 1], return_counts=True)
                s_vec[i] += w * counts_i  # i wins/loses (1/-1)
                s_vec[j] += -w * counts_j  # j loses/wins (1/-1)
            # print(s_vec)
            # print("\n")
            # Normalize S-vector
            if np.count_nonzero(s_vec) > 0:
                s_vec[s_vec > 0] = 1  # Normalize net wins
                s_vec[s_vec < 0] = -1  # Normalize net loses

                lst_gis.append(GI(col, '+'))
                cum_wins.append(temp_cum_wins)
                s_mat.append(s_vec)

                lst_gis.append(GI(col, '-'))
                cum_wins.append(-temp_cum_wins)
                s_mat.append(-s_vec)

        return np.array(lst_gis), np.array(cum_wins), np.array(s_mat), pair_ij

    def _construct_all_matrices(self):
        """Description

        Generates all the gradual items and, constructs: (1) win matrix (2) net-win matrix, (3) cumulative wins,
        (4) pairwise objects.

        :return: list of gradual items, win matrix, net-win matrix, cumulative win matrix, selected (ij) objects
        """

        n = self.row_count

        # 1a. Generate all possible pairs
        pair_ij = np.array(np.meshgrid(np.arange(n), np.arange(n))).T.reshape(-1, 2)

        # 1b. Remove duplicates or reversed pairs
        pair_ij = pair_ij[np.argwhere(pair_ij[:, 0] < pair_ij[:, 1])[:, 0]]

        # 2. Variable declarations
        attr_data = self.data.T  # Feature data objects
        lst_gis = []  # List of GIs
        s_mat = []  # S-Matrix (made up of S-Vectors)
        w_mat = []  # win matrix
        cum_wins = []  # Cumulative wins

        # 3. Construct S matrix from data set
        for col in np.nditer(self.attr_cols):
            # Feature data objects
            col_data = np.array(attr_data[col], dtype=np.float)  # Feature data objects

            # Cumulative Wins: for estimation of score-vector
            temp_cum_wins = np.where(col_data[pair_ij[:, 0]] < col_data[pair_ij[:, 1]], 1,
                                     np.where(col_data[pair_ij[:, 0]] > col_data[pair_ij[:, 1]], -1, 0))

            # S-vector
            s_vec = np.zeros((n,), dtype=np.int32)
            for w in [1, -1]:
                positions = np.flatnonzero(temp_cum_wins == w)
                i, counts_i = np.unique(pair_ij[positions, 0], return_counts=True)
                j, counts_j = np.unique(pair_ij[positions, 1], return_counts=True)
                s_vec[i] += w * counts_i  # i wins/loses (1/-1)
                s_vec[j] += -w * counts_j  # j loses/wins (1/-1)

            # Normalize S-vector
            if np.count_nonzero(s_vec) > 0:
                w_mat.append(np.copy(s_vec))

                s_vec[s_vec > 0] = 1  # Normalize net wins
                s_vec[s_vec < 0] = -1  # Normalize net loses

                lst_gis.append(GI(col, '+'))
                cum_wins.append(temp_cum_wins)
                s_mat.append(s_vec)

                lst_gis.append(GI(col, '-'))
                cum_wins.append(-temp_cum_wins)
                s_mat.append(-s_vec)

        return np.array(lst_gis), np.array(w_mat), np.array(cum_wins), np.array(s_mat), pair_ij

    def infer_gps(self, clusters):
        """Description

        A function that infers GPs from clusters of gradual items.

        :param clusters: [required] groups of gradual items clustered through K-MEANS algorithm
        :return: list of (str) patterns, list of GP objects
        """

        patterns = []
        str_patterns = []

        all_gis = self.gradual_items
        cum_wins = self.cum_wins

        lst_indices = [np.where(clusters == element)[0] for element in np.unique(clusters)]
        for grp_idx in lst_indices:
            if grp_idx.size > 1:
                # 1. Retrieve all cluster-pairs and the corresponding GIs
                cluster_gis = all_gis[grp_idx]
                cluster_cum_wins = cum_wins[grp_idx]  # All the rows of selected groups

                # 2. Compute score vector from R matrix
                score_vectors = []  # Approach 2
                for c_win in cluster_cum_wins:
                    temp = self._estimate_score_vector(c_win)
                    score_vectors.append(temp)

                # 3. Estimate support
                est_sup = self._estimate_support(score_vectors)

                # 4. Infer GPs from the clusters
                if est_sup >= self.thd_supp:
                    gp = ExtGP()
                    for gi in cluster_gis:
                        gp.add_gradual_item(gi)
                    gp.set_support(est_sup)
                    patterns.append(gp)
                    str_patterns.append(gp.print(self.titles))
        return str_patterns, patterns

    def _estimate_score_vector(self, c_wins):
        """Description

        A function that estimates the score vector based on the cumulative wins.

        :param c_wins: [required] cumulative wins
        :return: score vector (ndarray)
        """

        # Estimate score vector from pairs
        n = self.row_count
        score_vector = np.ones(shape=(n,))
        arr_ij = self.ij

        # Construct a win-matrix
        temp_vec = np.zeros(shape=(n,))
        pair_count = arr_ij.shape[0]

        # Compute score vector
        for k in range(self.max_iteration):
            if np.count_nonzero(score_vector == 0) > 1:
                break
            else:
                for pr in range(pair_count):
                    pr_val = c_wins[pr]
                    i = arr_ij[pr][0]
                    j = arr_ij[pr][1]
                    if pr_val == 1:
                        log = math.log(
                            math.exp(score_vector[i]) / (math.exp(score_vector[i]) + math.exp(score_vector[j])),
                            10)
                        temp_vec[i] += pr_val * log
                    elif pr_val == -1:
                        log = math.log(
                            math.exp(score_vector[j]) / (math.exp(score_vector[i]) + math.exp(score_vector[j])),
                            10)
                        temp_vec[j] += -pr_val * log
                score_vector = abs(temp_vec / np.sum(temp_vec))
        return score_vector

    def _estimate_support(self, score_vectors):
        """Description

        A function that estimates the frequency support of a GP based on its score vector.

        :param score_vectors: score vector (ndarray)
        :return: estimated support (float)
        """

        # Estimate support - use different score-vectors to construct pairs
        n = self.row_count
        bin_mat = np.ones((n, n), dtype=np.bool)
        for vec in score_vectors:
            temp_bin = vec < vec[:, np.newaxis]
            bin_mat = np.multiply(bin_mat, temp_bin)

        est_sup = float(np.sum(bin_mat)) / float(n * (n - 1.0) / 2.0)
        """:type est_sup: float"""
        return est_sup

    def discover(self, testing=False):
        """Description

        Applies spectral clustering to determine which gradual items belong to the same group based on the similarity
        of net-win vectors. Gradual items in the same cluster should have almost similar score vector. The candidates
        are validated if their computed support is greater than or equal to the minimum support threshold specified by
        the user.

        :param testing: [optional] returns different format if algorithm is used in a test environment
        :return: JSON object
        """

        # 1. Generate net-win matrices
        s_matrix = self.net_win_mat  # Net-win matrix (S)
        if s_matrix.size < 1:
            raise Exception("Erasure probability is too high, consider reducing it.")
        # print(s_matrix)

        start = time.time()  # TO BE REMOVED
        # 2a. Spectral Clustering: perform SVD to determine the independent rows
        u, s, vt = np.linalg.svd(s_matrix)

        # 2b. Spectral Clustering: compute rank of net-wins matrix
        r = np.linalg.matrix_rank(s_matrix)  # approximated r

        # 2c. Spectral Clustering: rank approximation
        s_matrix_approx = u[:, :r] @ np.diag(s[:r]) @ vt[:r, :]

        # 2d. Clustering using K-Means (using sklearn library)
        kmeans = KMeans(n_clusters=r, random_state=0)
        y_predicted = kmeans.fit_predict(s_matrix_approx)

        end = time.time()  # TO BE REMOVED

        # 3. Infer GPs
        str_gps, estimated_gps = self.infer_gps(y_predicted)

        # 4. Output - DO NOT ADD TO PyPi Package
        out = structure()
        out.estimated_gps = estimated_gps
        out.max_iteration = self.max_iteration
        out.titles = self.titles
        out.col_count = self.col_count
        out.row_count = self.row_count
        out.e_prob = self.erasure_probability
        out.cluster_time = (end - start)  # TO BE REMOVED
        if testing:
            return out

        # Output
        out = json.dumps({"Algorithm": "Clu-GRAANK", "Patterns": str_gps, "Invalid Count": 0})
        """:type out: object"""
        self.gradual_patterns = estimated_gps
        return out


class GeneticGRAANK(DataGP):
    """Description

    Extract gradual patterns (GPs) from a numeric data source using the Genetic Algorithm approach (proposed
    in a published  paper by Dickson Owuor). A GP is a set of gradual items (GI) and its quality is measured by
    its computed support value. For example given a data set with 3 columns (age, salary, cars) and 10 objects.
    A GP may take the form: {age+, salary-} with a support of 0.8. This implies that 8 out of 10 objects have the
    values of column age 'increasing' and column 'salary' decreasing.

         In this approach, we assume that every GP candidate may be represented as a binary gene (or individual) that
         has a unique position and cost. The cost is derived from the computed support of that candidate, the higher the
         support value the lower the cost. The aim of the algorithm is search through a population of individuals (or
         candidates) and find those with the lowest cost as efficiently as possible.

    This class extends class DataGP, and it provides the following additional attributes:

        max_iteration: integer value determines the number of iterations for the algorithm

        n_pop: integer value that determines the initial population size of individuals

        pc: a value that determines the proportion of children

        gamma: a value in the range 0-1 that determines the cross-over rate

        mu: a value in the range 0-1 that determines the mutation rate

        sigma: a value in the range 0-1 that determines the mutation rate

    """

    def __init__(self, *args, max_iter=MAX_ITERATIONS, n_pop=N_POPULATION, pc=PC, gamma=GAMMA, mu=MU, sigma=SIGMA):
        """Description

        Extract gradual patterns (GPs) from a numeric data source using the Genetic Algorithm approach (proposed
        in a published  paper by Dickson Owuor). A GP is a set of gradual items (GI) and its quality is measured by
        its computed support value. For example given a data set with 3 columns (age, salary, cars) and 10 objects.
        A GP may take the form: {age+, salary-} with a support of 0.8. This implies that 8 out of 10 objects have the
        values of column age 'increasing' and column 'salary' decreasing.

             In this approach, we assume that every GP candidate may be represented as a binary gene (or individual)
             that has a unique position and cost. The cost is derived from the computed support of that candidate, the
             higher the support value the lower the cost. The aim of the algorithm is search through a population of
             individuals (or candidates) and find those with the lowest cost as efficiently as possible.

        This class extends class DataGP, and it provides the following additional attributes:

            max_iteration: integer value determines the number of iterations for the algorithm

            n_pop: integer value that determines the initial population size of individuals

            pc: a value that determines the proportion of children

            gamma: a value in the range 0-1 that determines the cross-over rate

            mu: a value in the range 0-1 that determines the mutation rate

            sigma: a value in the range 0-1 that determines the mutation rate


        :param args: [required] data-source, [optional] minimum-support
        :param max_iter: maximum_iteration, default is 1
        :param n_pop: initial individual population, default is 5
        :param pc: children proportion, default is 0.5
        :param gamma: cross-over gamma ratio, default is 1
        :param mu: mutation mu ratio, default is 0.9
        :param sigma: mutation sigma ratio, default is 0.9
        """
        super(GeneticGRAANK, self).__init__(*args)
        self.max_iteration = max_iter
        self.n_pop = n_pop
        self.pc = pc
        self.gamma = gamma
        self.mu = mu
        self.sigma = sigma

    def _crossover(self, p1, p2):
        """Description

        Crosses over the genes of 2 parents (an individual with a specific position and cost) in order to generate 2
        different offsprings.

        :param p1: parent 1 individual
        :param p2: parent 2 individual
        :return: 2 offsprings (children)
        """
        c1 = p1.deepcopy()
        c2 = p2.deepcopy()
        alpha = np.random.uniform(0, self.gamma, 1)
        c1.position = alpha * p1.position + (1 - alpha) * p2.position
        c2.position = alpha * p2.position + (1 - alpha) * p1.position
        return c1, c2

    def _mutate(self, x):
        """Description

        Mutates an individual's position in order to create a new and different individual.

        :param x: existing individual
        :return: new individual
        """
        y = x.deepcopy()
        str_x = str(int(y.position))
        flag = np.random.rand(*(len(str_x),)) <= self.mu
        ind = np.argwhere(flag)
        str_y = "0"
        for i in ind:
            val = float(str_x[i[0]])
            val += self.sigma * np.random.uniform(0, 1, 1)
            if i[0] == 0:
                str_y = "".join(("", "{}".format(int(val)), str_x[1:]))
            else:
                str_y = "".join((str_x[:i[0] - 1], "{}".format(int(val)), str_x[i[0]:]))
            str_x = str_y
        y.position = int(str_y)
        return y

    def discover(self):
        """Description

        Uses genetic algorithm to find GP candidates. The candidates are validated if their computed support is greater
        than or equal to the minimum support threshold specified by the user.

        :return: JSON object
        """

        # Prepare data set
        self.fit_bitmap()
        attr_keys = [GI(x[0], x[1].decode()).as_string() for x in self.valid_bins[:, 0]]

        if self.no_bins:
            return []

        # Problem Information
        # cost_function

        # Parameters
        # pc: Proportion of children (if its 1, then nc == npop
        it_count = 0
        eval_count = 0
        counter = 0
        var_min = 0
        var_max = int(''.join(['1'] * len(attr_keys)), 2)

        nc = int(np.round(self.pc * self.n_pop / 2) * 2)  # No. of children np.round is used to get even number

        # Empty Individual Template
        empty_individual = structure()
        empty_individual.position = None
        empty_individual.cost = None

        # Initialize Population
        pop = empty_individual.repeat(self.n_pop)
        for i in range(self.n_pop):
            pop[i].position = random.randrange(var_min, var_max)
            pop[i].cost = 1  # cost_function(pop[i].position, attr_keys, d_set)
            # if pop[i].cost < best_sol.cost:
            #    best_sol = pop[i].deepcopy()

        # Best Solution Ever Found
        best_sol = empty_individual.deepcopy()
        best_sol.position = pop[0].position
        best_sol.cost = NumericSS.cost_function(best_sol.position, attr_keys, self)

        # Best Cost of Iteration
        best_costs = np.empty(self.max_iteration)
        best_patterns = list()
        str_best_gps = list()
        str_iter = ''
        str_eval = ''

        invalid_count = 0
        repeated = 0

        while counter < self.max_iteration:
            # while eval_count < max_evaluations:
            # while repeated < 1:

            c_pop = []  # Children population
            for _ in range(nc // 2):
                # Select Parents
                q = np.random.permutation(self.n_pop)
                p1 = pop[q[0]]
                p2 = pop[q[1]]

                # a. Perform Crossover
                c1, c2 = self._crossover(p1, p2)

                # Apply Bound
                NumericSS.apply_bound(c1, var_min, var_max)
                NumericSS.apply_bound(c2, var_min, var_max)

                # Evaluate First Offspring
                c1.cost = NumericSS.cost_function(c1.position, attr_keys, self)
                if c1.cost == 1:
                    invalid_count += 1
                if c1.cost < best_sol.cost:
                    best_sol = c1.deepcopy()
                eval_count += 1
                str_eval += "{}: {} \n".format(eval_count, best_sol.cost)

                # Evaluate Second Offspring
                c2.cost = NumericSS.cost_function(c2.position, attr_keys, self)
                if c1.cost == 1:
                    invalid_count += 1
                if c2.cost < best_sol.cost:
                    best_sol = c2.deepcopy()
                eval_count += 1
                str_eval += "{}: {} \n".format(eval_count, best_sol.cost)

                # b. Perform Mutation
                c1 = self._mutate(c1)
                c2 = self._mutate(c2)

                # Apply Bound
                NumericSS.apply_bound(c1, var_min, var_max)
                NumericSS.apply_bound(c2, var_min, var_max)

                # Evaluate First Offspring
                c1.cost = NumericSS.cost_function(c1.position, attr_keys, self)
                if c1.cost == 1:
                    invalid_count += 1
                if c1.cost < best_sol.cost:
                    best_sol = c1.deepcopy()
                eval_count += 1
                str_eval += "{}: {} \n".format(eval_count, best_sol.cost)

                # Evaluate Second Offspring
                c2.cost = NumericSS.cost_function(c2.position, attr_keys, self)
                if c1.cost == 1:
                    invalid_count += 1
                if c2.cost < best_sol.cost:
                    best_sol = c2.deepcopy()
                eval_count += 1
                str_eval += "{}: {} \n".format(eval_count, best_sol.cost)

                # c. Add Offsprings to c_pop
                c_pop.append(c1)
                c_pop.append(c2)

            # Merge, Sort and Select
            pop += c_pop
            pop = sorted(pop, key=lambda x: x.cost)
            pop = pop[0:self.n_pop]

            best_gp = NumericSS.decode_gp(attr_keys, best_sol.position).validate_graank(self)
            """:type best_gp: ExtGP"""
            is_present = best_gp.is_duplicate(best_patterns)
            is_sub = best_gp.check_am(best_patterns, subset=True)
            if is_present or is_sub:
                repeated += 1
            else:
                if best_gp.support >= self.thd_supp:
                    best_patterns.append(best_gp)
                    str_best_gps.append(best_gp.print(self.titles))
                # else:
                #    best_sol.cost = 1

            try:
                # Show Iteration Information
                # Store Best Cost
                best_costs[it_count] = best_sol.cost
                str_iter += "{}: {} \n".format(it_count, best_sol.cost)
            except IndexError:
                pass
            it_count += 1

            if self.max_iteration == 1:
                counter = repeated
            else:
                counter = it_count
        # Output
        out = json.dumps({"Algorithm": "GA-GRAANK", "Best Patterns": str_best_gps, "Invalid Count": invalid_count,
                          "Iterations": it_count})
        """:type out: object"""
        self.gradual_patterns = best_patterns
        return out


class GRAANK(DataGP):
    """Description

    Extracts gradual patterns (GPs) from a numeric data source using the GRAANK approach (proposed in a published
    research paper by Anne Laurent).

         A GP is a set of gradual items (GI) and its quality is measured by its computed support value. For example
         given a data set with 3 columns (age, salary, cars) and 10 objects. A GP may take the form: {age+, salary-}
         with a support of 0.8. This implies that 8 out of 10 objects have the values of column age 'increasing' and
         column 'salary' decreasing.

    This class extends class DataGP which is responsible for generating the GP bitmaps.

    """

    def _gen_apriori_candidates(self, gi_bins):
        """Description

        Generates Apriori GP candidates
        :param gi_bins: GI together with bitmaps
        :return:
        """
        sup = self.thd_supp
        n = self.row_count

        invalid_count = 0
        res = []
        all_candidates = []
        if len(gi_bins) < 2:
            return []
        try:
            set_gi = [{x[0]} for x in gi_bins]
        except TypeError:
            set_gi = [set(x[0]) for x in gi_bins]

        for i in range(len(gi_bins) - 1):
            for j in range(i + 1, len(gi_bins)):
                try:
                    gi_i = {gi_bins[i][0]}
                    gi_j = {gi_bins[j][0]}
                    gi_o = {gi_bins[0][0]}
                except TypeError:
                    gi_i = set(gi_bins[i][0])
                    gi_j = set(gi_bins[j][0])
                    gi_o = set(gi_bins[0][0])
                gp_cand = gi_i | gi_j
                inv_gp_cand = {GI.inv_arr(x) for x in gp_cand}
                if (len(gp_cand) == len(gi_o) + 1) and (not (all_candidates != [] and gp_cand in all_candidates)) \
                        and (not (all_candidates != [] and inv_gp_cand in all_candidates)):
                    test = 1
                    for k in gp_cand:
                        try:
                            k_set = {k}
                        except TypeError:
                            k_set = set(k)
                        gp_cand_2 = gp_cand - k_set
                        inv_gp_cand_2 = {GI.inv_arr(x) for x in gp_cand_2}
                        if gp_cand_2 not in set_gi and inv_gp_cand_2 not in set_gi:
                            test = 0
                            break
                    if test == 1:
                        m = gi_bins[i][1] * gi_bins[j][1]
                        t = float(np.sum(m)) / float(n * (n - 1.0) / 2.0)
                        if t > sup:
                            res.append([gp_cand, m])
                        else:
                            invalid_count += 1
                    all_candidates.append(gp_cand)
                    gc.collect()
        return res, invalid_count

    def discover(self):
        """Description

        Uses apriori algorithm to find GP candidates. The candidates are validated if their computed support is greater
        than or equal to the minimum support threshold specified by the user.

        :return: JSON object
        """

        self.fit_bitmap()

        self.gradual_patterns = []
        """:type patterns: GP list"""
        str_winner_gps = []
        n = self.attr_size
        valid_bins = self.valid_bins

        invalid_count = 0
        while len(valid_bins) > 0:
            valid_bins, inv_count = self._gen_apriori_candidates(valid_bins)
            invalid_count += inv_count
            i = 0
            while i < len(valid_bins) and valid_bins != []:
                gi_tuple = valid_bins[i][0]
                bin_data = valid_bins[i][1]
                sup = float(np.sum(np.array(bin_data))) / float(n * (n - 1.0) / 2.0)
                if sup < self.thd_supp:
                    del valid_bins[i]
                    invalid_count += 1
                else:
                    z = 0
                    while z < (len(self.gradual_patterns) - 1):
                        if set(self.gradual_patterns[z].get_pattern()).issubset(set(gi_tuple)):
                            del self.gradual_patterns[z]
                        else:
                            z = z + 1

                    gp = GP()
                    """:type gp: GP"""
                    for obj in valid_bins[i][0]:
                        gi = GI(obj[0], obj[1].decode())
                        """:type gi: GI"""
                        gp.add_gradual_item(gi)
                    gp.set_support(sup)
                    self.gradual_patterns.append(gp)
                    str_winner_gps.append(gp.print(self.titles))
                    i += 1
        # Output
        out = json.dumps({"Algorithm": "GRAANK", "Patterns": str_winner_gps, "Invalid Count": invalid_count})
        """:type out: object"""
        return out


class HillClimbingGRAANK(DataGP):
    """Description

    Extract gradual patterns (GPs) from a numeric data source using the Hill Climbing (Local Search) Algorithm
    approach (proposed in a published research paper by Dickson Owuor). A GP is a set of gradual items (GI) and its
    quality is measured by its computed support value. For example given a data set with 3 columns (age, salary,
    cars) and 10 objects. A GP may take the form: {age+, salary-} with a support of 0.8. This implies that 8 out of
    10 objects have the values of column age 'increasing' and column 'salary' decreasing.

         In this approach, it is assumed that every GP candidate may be represented as a position that has a cost value
         associated with it. The cost is derived from the computed support of that candidate, the higher the support
         value the lower the cost. The aim of the algorithm is search through group of positions and find those with
         the lowest cost as efficiently as possible.

    This class extends class DataGP, and it provides the following additional attributes:

        max_iteration: integer value determines the number of iterations for the algorithm

        step_size: integer value that steps the algorithm takes per iteration

    """

    def __init__(self, *args, max_iter=MAX_ITERATIONS, step_size=STEP_SIZE):
        """Description

        Extract gradual patterns (GPs) from a numeric data source using the Hill Climbing (Local Search) Algorithm
        approach (proposed in a published research paper by Dickson Owuor). A GP is a set of gradual items (GI) and its
        quality is measured by its computed support value. For example given a data set with 3 columns (age, salary,
        cars) and 10 objects. A GP may take the form: {age+, salary-} with a support of 0.8. This implies that 8 out of
        10 objects have the values of column age 'increasing' and column 'salary' decreasing.

             In this approach, we assume that every GP candidate may be represented as a position that has cost value
             associated with it. The cost is derived from the computed support of that candidate, the higher the support
             value the lower the cost. The aim of the algorithm is search through group of positions and find those with
             the lowest cost as efficiently as possible.

        This class extends class DataGP, and it provides the following additional attributes:

            max_iteration: integer value determines the number of iterations for the algorithm

            step_size: integer value that steps the algorithm takes per iteration

        :param args: [required] data-source, [optional] minimum-support
        :param max_iter: maximum_iteration, default is 1
        :param step_size: step size, default is 0.5
        """
        super(HillClimbingGRAANK, self).__init__(*args)
        self.step_size = step_size
        self.max_iteration = max_iter

    def discover(self):
        """Description

        Uses hill-climbing algorithm to find GP candidates. The candidates are validated if their computed support is
        greater than or equal to the minimum support threshold specified by the user.

        :return: JSON object
        """
        # Prepare data set
        self.fit_bitmap()
        attr_keys = [GI(x[0], x[1].decode()).as_string() for x in self.valid_bins[:, 0]]

        if self.no_bins:
            return []

        # Parameters
        it_count = 0
        var_min = 0
        counter = 0
        var_max = int(''.join(['1'] * len(attr_keys)), 2)
        eval_count = 0

        # Empty Individual Template
        best_sol = structure()
        candidate = structure()

        # generate an initial point
        best_sol.position = None
        # candidate.position = None
        if best_sol.position is None:
            best_sol.position = np.random.uniform(var_min, var_max, N_VAR)
        # evaluate the initial point
        NumericSS.apply_bound(best_sol, var_min, var_max)
        best_sol.cost = NumericSS.cost_function(best_sol.position, attr_keys, self)

        # Best Cost of Iteration
        best_costs = np.empty(self.max_iteration)
        best_patterns = []
        str_best_gps = list()
        str_iter = ''
        str_eval = ''

        invalid_count = 0
        repeated = 0

        # run the hill climb
        while counter < self.max_iteration:
            # while eval_count < max_evaluations:
            # take a step
            candidate.position = None
            if candidate.position is None:
                candidate.position = best_sol.position + (random.randrange(var_min, var_max) * self.step_size)
            NumericSS.apply_bound(candidate, var_min, var_max)
            candidate.cost = NumericSS.cost_function(candidate.position, attr_keys, self)
            if candidate.cost == 1:
                invalid_count += 1

            if candidate.cost < best_sol.cost:
                best_sol = candidate.deepcopy()
            eval_count += 1
            str_eval += "{}: {} \n".format(eval_count, best_sol.cost)

            best_gp = NumericSS.decode_gp(attr_keys, best_sol.position).validate_graank(self)
            """:type best_gp: ExtGP"""
            is_present = best_gp.is_duplicate(best_patterns)
            is_sub = best_gp.check_am(best_patterns, subset=True)
            if is_present or is_sub:
                repeated += 1
            else:
                if best_gp.support >= self.thd_supp:
                    best_patterns.append(best_gp)
                    str_best_gps.append(best_gp.print(self.titles))

            try:
                # Show Iteration Information
                # Store Best Cost
                best_costs[it_count] = best_sol.cost
                str_iter += "{}: {} \n".format(it_count, best_sol.cost)
            except IndexError:
                pass
            it_count += 1

            if self.max_iteration == 1:
                counter = repeated
            else:
                counter = it_count
        # Output
        out = json.dumps({"Algorithm": "LS-GRAANK", "Best Patterns": str_best_gps, "Invalid Count": invalid_count,
                          "Iterations": it_count})
        """:type out: object"""
        self.gradual_patterns = best_patterns
        return out


class ParticleGRAANK(DataGP):
    """Description

    Extract gradual patterns (GPs) from a numeric data source using the Particle Swarm Optimization Algorithm
    approach (proposed in a published research paper by Dickson Owuor). A GP is a set of gradual items (GI) and its
    quality is measured by its computed support value. For example given a data set with 3 columns (age, salary,
    cars) and 10 objects. A GP may take the form: {age+, salary-} with a support of 0.8. This implies that 8 out of
    10 objects have the values of column age 'increasing' and column 'salary' decreasing.

         In this approach, it is assumed that every GP candidate may be represented as a particle that has a unique
         position and fitness. The fitness is derived from the computed support of that candidate, the higher the
         support value the higher the fitness. The aim of the algorithm is search through a population of particles
         (or candidates) and find those with the highest fitness as efficiently as possible.

    This class extends class DataGP, and it provides the following additional attributes:

        max_iteration: integer value determines the number of iterations for the algorithm

        n_particle: integer value that determines the initial population size of particles

        vel: a value that determines the velocity of particles

        coeff_p: a value in the range 0-1, personal coefficient

        coeff_g: a value in the range 0-1, global coefficient


    """

    def __init__(self, *args, max_iter=MAX_ITERATIONS, n_particle=N_PARTICLES, vel=VELOCITY, coeff_p=PERSONAL_COEFF,
                 coeff_g=GLOBAL_COEFF):
        """Description

        Extract gradual patterns (GPs) from a numeric data source using the Particle Swarm Optimization Algorithm
        approach (proposed in a published research paper by Dickson Owuor). A GP is a set of gradual items (GI) and its
        quality is measured by its computed support value. For example given a data set with 3 columns (age, salary,
        cars) and 10 objects. A GP may take the form: {age+, salary-} with a support of 0.8. This implies that 8 out of
        10 objects have the values of column age 'increasing' and column 'salary' decreasing.

            In this approach, it is assumed that every GP candidate may be represented as a particle that has a unique
            position and fitness. The fitness is derived from the computed support of that candidate, the higher the
            support value the higher the fitness. The aim of the algorithm is search through a population of particles
            (or candidates) and find those with the highest fitness as efficiently as possible.

        This class extends class DataGP, and it provides the following additional attributes:

            max_iteration: integer value determines the number of iterations for the algorithm

            n_particle: integer value that determines the initial population size of particles

            vel: a value that determines the velocity of particles

            coeff_p: a value in the range 0-1, personal coefficient

            coeff_g: a value in the range 0-1, global coefficient

        :param args: [required] data-source, [optional] minimum-support
        :param max_iter: maximum_iteration, default is 1
        :param n_particle: initial particle population, default is 5
        :param vel: velocity, default is 0.9
        :param coeff_p: personal coefficient, default is 0.01
        :param coeff_g: global coefficient, default is 0.9
        """
        super(ParticleGRAANK, self).__init__(*args)
        self.max_iteration = max_iter
        self.n_particles = n_particle
        self.velocity = vel
        self.coeff_p = coeff_p
        self.coeff_g = coeff_g

    def discover(self):
        """Description

        Searches through particle positions to find GP candidates. The candidates are validated if their computed
        support is greater than or equal to the minimum support threshold specified by the user.

        :return: JSON object
        """

        # Prepare data set
        self.fit_bitmap()

        # self.target = 1
        # self.target_error = 1e-6
        attr_keys = [GI(x[0], x[1].decode()).as_string() for x in self.valid_bins[:, 0]]

        if self.no_bins:
            return []

        it_count = 0
        eval_count = 0
        counter = 0
        var_min = 0
        var_max = int(''.join(['1'] * len(attr_keys)), 2)

        # Empty particle template
        empty_particle = structure()
        empty_particle.position = None
        empty_particle.fitness = None

        # Initialize Population
        particle_pop = empty_particle.repeat(self.n_particles)
        for i in range(self.n_particles):
            particle_pop[i].position = random.randrange(var_min, var_max)
            particle_pop[i].fitness = 1

        pbest_pop = particle_pop.copy()
        gbest_particle = pbest_pop[0]

        # Best particle (ever found)
        best_particle = empty_particle.deepcopy()
        best_particle.position = gbest_particle.position
        best_particle.fitness = NumericSS.cost_function(best_particle.position, attr_keys, self)

        velocity_vector = np.ones(self.n_particles)
        best_fitness_arr = np.empty(self.max_iteration)
        best_patterns = []
        str_best_gps = list()
        str_iter = ''
        str_eval = ''

        invalid_count = 0
        repeated = 0

        while counter < self.max_iteration:
            # while eval_count < max_evaluations:
            # while repeated < 1:
            for i in range(self.n_particles):
                # UPDATED
                if particle_pop[i].position < var_min or particle_pop[i].position > var_max:
                    particle_pop[i].fitness = 1
                else:
                    particle_pop[i].fitness = NumericSS.cost_function(particle_pop[i].position, attr_keys, self)
                    if particle_pop[i].fitness == 1:
                        invalid_count += 1
                    eval_count += 1
                    str_eval += "{}: {} \n".format(eval_count, particle_pop[i].fitness)

                if pbest_pop[i].fitness > particle_pop[i].fitness:
                    pbest_pop[i].fitness = particle_pop[i].fitness
                    pbest_pop[i].position = particle_pop[i].position

                if gbest_particle.fitness > particle_pop[i].fitness:
                    gbest_particle.fitness = particle_pop[i].fitness
                    gbest_particle.position = particle_pop[i].position
            # if abs(gbest_fitness_value - self.target) < self.target_error:
            #    break
            if best_particle.fitness > gbest_particle.fitness:
                best_particle = gbest_particle.deepcopy()

            for i in range(self.n_particles):
                new_velocity = (self.velocity * velocity_vector[i]) + \
                               (self.coeff_p * random.random()) * (pbest_pop[i].position - particle_pop[i].position) + \
                               (self.coeff_g * random.random()) * (gbest_particle.position - particle_pop[i].position)
                particle_pop[i].position = particle_pop[i].position + new_velocity

            best_gp = NumericSS.decode_gp(attr_keys, best_particle.position).validate_graank(self)
            """:type best_gp: ExtGP"""
            is_present = best_gp.is_duplicate(best_patterns)
            is_sub = best_gp.check_am(best_patterns, subset=True)
            if is_present or is_sub:
                repeated += 1
            else:
                if best_gp.support >= self.thd_supp:
                    best_patterns.append(best_gp)
                    str_best_gps.append(best_gp.print(self.titles))
                # else:
                #    best_particle.fitness = 1

            try:
                # Show Iteration Information
                best_fitness_arr[it_count] = best_particle.fitness
                str_iter += "{}: {} \n".format(it_count, best_particle.fitness)
            except IndexError:
                pass
            it_count += 1

            if self.max_iteration == 1:
                counter = repeated
            else:
                counter = it_count
        # Output
        out = json.dumps({"Algorithm": "PSO-GRAANK", "Best Patterns": str_best_gps, "Invalid Count": invalid_count,
                          "Iterations": it_count})
        """:type out: object"""
        self.gradual_patterns = best_patterns

        return out


class RandomGRAANK(DataGP):
    """Description

    Extract gradual patterns (GPs) from a numeric data source using the Random Search Algorithm (LS-GRAANK)
    approach (proposed in a published research paper by Dickson Owuor). A GP is a set of gradual items (GI) and its
    quality is measured by its computed support value. For example given a data set with 3 columns (age, salary,
    cars) and 10 objects. A GP may take the form: {age+, salary-} with a support of 0.8. This implies that 8 out of
    10 objects have the values of column age 'increasing' and column 'salary' decreasing.

         In this approach, it is assumed that every GP candidate may be represented as a position that has a cost value
         associated with it. The cost is derived from the computed support of that candidate, the higher the support
         value the lower the cost. The aim of the algorithm is search through group of positions and find those with
         the lowest cost as efficiently as possible.

    This class extends class DataGP, and it provides the following additional attributes:

        max_iteration: integer value determines the number of iterations for the algorithm

    """

    def __init__(self, *args, max_iter=MAX_ITERATIONS):
        """Description

        Extract gradual patterns (GPs) from a numeric data source using the Random Search Algorithm (LS-GRAANK)
        approach (proposed in a published research paper by Dickson Owuor). A GP is a set of gradual items (GI) and its
        quality is measured by its computed support value. For example given a data set with 3 columns (age, salary,
        cars) and 10 objects. A GP may take the form: {age+, salary-} with a support of 0.8. This implies that 8 out of
        10 objects have the values of column age 'increasing' and column 'salary' decreasing.

            In this approach, we assume that every GP candidate may be represented as a position that has a cost value
            associated with it. The cost is derived from the computed support of that candidate, the higher the support
            value the lower the cost. The aim of the algorithm is search through group of positions and find those with
            the lowest cost as efficiently as possible.

        This class extends class DataGP, and it provides the following additional attributes:

            max_iteration: integer value determines the number of iterations for the algorithm

        :param args: [required] data-source, [optional] minimum-support
        :param max_iter: maximum_iteration, default is 1
        """
        super(RandomGRAANK, self).__init__(*args)
        self.max_iteration = max_iter

    def discover(self):
        """Description

        Uses random search to find GP candidates. The candidates are validated if their computed support is greater
        than or equal to the minimum support threshold specified by the user.

        :return: JSON object
        """
        # Prepare data set
        self.fit_bitmap()
        attr_keys = [GI(x[0], x[1].decode()).as_string() for x in self.valid_bins[:, 0]]

        if self.no_bins:
            return []

        # Parameters
        it_count = 0
        counter = 0
        var_min = 0
        var_max = int(''.join(['1'] * len(attr_keys)), 2)
        eval_count = 0

        # Empty Individual Template
        candidate = structure()
        candidate.position = None
        candidate.cost = float('inf')

        # INITIALIZE
        best_sol = candidate.deepcopy()
        best_sol.position = np.random.uniform(var_min, var_max, N_VAR)
        best_sol.cost = NumericSS.cost_function(best_sol.position, attr_keys, self)

        # Best Cost of Iteration
        best_costs = np.empty(self.max_iteration)
        best_patterns = []
        str_best_gps = list()
        str_iter = ''
        str_eval = ''

        repeated = 0
        invalid_count = 0

        while counter < self.max_iteration:
            # while eval_count < max_evaluations:
            candidate.position = ((var_min + random.random()) * (var_max - var_min))
            NumericSS.apply_bound(candidate, var_min, var_max)
            candidate.cost = NumericSS.cost_function(candidate.position, attr_keys, self)
            if candidate.cost == 1:
                invalid_count += 1

            if candidate.cost < best_sol.cost:
                best_sol = candidate.deepcopy()
            eval_count += 1
            str_eval += "{}: {} \n".format(eval_count, best_sol.cost)

            best_gp = NumericSS.decode_gp(attr_keys, best_sol.position).validate_graank(self)
            """:type best_gp: ExtGP"""
            is_present = best_gp.is_duplicate(best_patterns)
            is_sub = best_gp.check_am(best_patterns, subset=True)
            if is_present or is_sub:
                repeated += 1
            else:
                if best_gp.support >= self.thd_supp:
                    best_patterns.append(best_gp)
                    str_best_gps.append(best_gp.print(self.titles))
                # else:
                #    best_sol.cost = 1

            try:
                # Show Iteration Information
                # Store Best Cost
                best_costs[it_count] = best_sol.cost
                str_iter += "{}: {} \n".format(it_count, best_sol.cost)
            except IndexError:
                pass
            it_count += 1

            if self.max_iteration == 1:
                counter = repeated
            else:
                counter = it_count
        # Output
        out = json.dumps({"Algorithm": "RS-GRAANK", "Best Patterns": str_best_gps, "Invalid Count": invalid_count,
                          "Iterations": it_count})
        """:type out: object"""
        self.gradual_patterns = best_patterns
        return out
