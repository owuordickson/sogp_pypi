# -*- coding: utf-8 -*-
"""
@author: "Dickson Owuor"
@credits: "Thomas Runkler, Edmond Menya, and Anne Laurent,"
@license: "MIT"
@version: "0.0.1"
@email: "owuordickson@gmail.com"
@created: "21 July 2021"
@modified: "22 July 2021"

SWARM OPTIMIZATION FOR GRADUAL PATTERNS:
Swarm optimization algorithms for mining gradual patterns


"""

import csv
from dateutil.parser import parse
import time
import gc
import numpy as np
import json
import pandas as pd


# -------- CONFIGURATION (START)-------------

# Global Swarm Configurations
MIN_SUPPORT = 0.5
MAX_ITERATIONS = 1
N_VAR = 1  # DO NOT CHANGE

# ACO-GRAD Configurations:
EVAPORATION_FACTOR = 0.5

# GA-GRAD Configurations:
N_POPULATION = 5
PC = 0.5
GAMMA = 1  # Cross-over
MU = 0.9  # Mutation
SIGMA = 0.9  # Mutation

# PSO-GRAD Configurations:
VELOCITY = 0.9  # higher values helps to move to next number in search space
PERSONAL_COEFF = 0.01
GLOBAL_COEFF = 0.9
TARGET = 1
TARGET_ERROR = 1e-6
N_PARTICLES = 5

# PLS-GRAD Configurations
STEP_SIZE = 0.5

# VISUALIZATIONS
SHOW_P_MATRIX = False  # ONLY FOR: aco
SHOW_EVALUATIONS = False  # FOR aco, prs, pls, pso
SHOW_ITERATIONS = True  # FOR aco, prs, pls, pso
SAVE_RESULTS = True  # FOR aco, prs, pls, pso


# -------- CONFIGURATION (END)-------------


# -------- DATA SET PREPARATION (START)-------------

"""
Changes
-------
1. Fetch all binaries during initialization
2. Replaced loops for fetching binary rank with numpy function
3. Accepts CSV file as data source OR
4. Accepts Pandas DataFrame as data source
5. Cleans data set
"""


class Dataset:

    def __init__(self, data_source, min_sup=0.5, eq=False):
        self.thd_supp = min_sup
        self.equal = eq
        self.titles, self.data = Dataset.read(data_source)
        self.row_count, self.col_count = self.data.shape
        self.time_cols = self.get_time_cols()
        self.attr_cols = self.get_attr_cols()
        self.valid_bins = np.array([])
        self.no_bins = False
        self.step_name = ''  # For T-GRAANK
        self.attr_size = 0  # For T-GRAANK

    def get_attr_cols(self):
        all_cols = np.arange(self.col_count)
        attr_cols = np.setdiff1d(all_cols, self.time_cols)
        return attr_cols

    def get_time_cols(self):
        # Retrieve first column only
        time_cols = list()
        n = self.col_count
        for i in range(n):  # check every column/attribute for time format
            row_data = str(self.data[0][i])
            try:
                time_ok, t_stamp = Dataset.test_time(row_data)
                if time_ok:
                    time_cols.append(i)
            except ValueError:
                continue
        return np.array(time_cols)

    def init_gp_attributes(self, attr_data=None):
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
            # temp_pos = Dataset.bin_rank(col_data, equal=self.equal)

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
        # 1. Retrieve data set from source
        if isinstance(data_src, pd.DataFrame):
            # a. DataFrame source
            # d_frame = pd.read_csv(d_fram,sep=';')  # TO BE REMOVED
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
            return Dataset.clean_data(data_src)
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
                    print("Data fetched from CSV file")
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
                    # titles = np.rec.fromarrays((keys, values), names=('key', 'value'))
                    # return titles, np.asarray(raw_data)
                    d_frame = pd.DataFrame(raw_data, columns=header)
                    return Dataset.clean_data(d_frame)
            except Exception as error:
                raise Exception("Error: " + str(error))

    @staticmethod
    def test_time(date_str):
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
        # 1. Remove objects with Null values
        df = df.dropna()

        # 2. Remove columns with Strings
        cols_to_remove = []
        for col in df.columns:
            try:
                _ = df[col].astype(float)
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
        print("Data cleaned")
        return titles, df.values


# -------- DATA SET PREPARATION (END)-------------


# -------- GRADUAL PATTERNS (START)-------------

"""
@created: "20 May 2020"
@modified: "21 Jul 2021"

GI: Gradual Item (0, +)
GP: Gradual Pattern {(0, +), (1, -), (3, +)}

"""


class GI:

    def __init__(self, attr_col, symbol):
        self.attribute_col = attr_col
        self.symbol = symbol
        self.gradual_item = np.array((attr_col, symbol), dtype='i, S1')
        self.tuple = tuple([attr_col, symbol])
        self.rank_sum = 0

    def inv(self):
        if self.symbol == '+':
            # temp = tuple([self.attribute_col, '-'])
            temp = np.array((self.attribute_col, '-'), dtype='i, S1')
        elif self.symbol == '-':
            # temp = tuple([self.attribute_col, '+'])
            temp = np.array((self.attribute_col, '+'), dtype='i, S1')
        else:
            temp = np.array((self.attribute_col, 'x'), dtype='i, S1')
        return temp

    def as_integer(self):
        if self.symbol == '+':
            temp = [self.attribute_col, 1]
        elif self.symbol == '-':
            temp = [self.attribute_col, -1]
        else:
            temp = [self.attribute_col, 0]
        return temp

    def as_string(self):
        if self.symbol == '+':
            temp = str(self.attribute_col) + '_pos'
        elif self.symbol == '-':
            temp = str(self.attribute_col) + '_neg'
        else:
            temp = str(self.attribute_col) + '_inv'
        return temp

    def to_string(self):
        return str(self.attribute_col) + self.symbol

    def is_decrement(self):
        if self.symbol == '-':
            return True
        else:
            return False

    @staticmethod
    def parse_gi(gi_str):
        txt = gi_str.split('_')
        attr_col = int(txt[0])
        if txt[1] == 'neg':
            symbol = '-'
        else:
            symbol = '+'
        return GI(attr_col, symbol)


class GP:

    def __init__(self):
        self.gradual_items = list()
        self.support = 0

    def set_support(self, support):
        self.support = round(support, 3)

    def add_gradual_item(self, item):
        if item.symbol == '-' or item.symbol == '+':
            self.gradual_items.append(item)
        else:
            pass

    def get_pattern(self):
        pattern = list()
        for item in self.gradual_items:
            pattern.append(item.gradual_item.tolist())
        return pattern

    def get_np_pattern(self):
        pattern = []
        for item in self.gradual_items:
            pattern.append(item.gradual_item)
        return np.array(pattern)

    def get_tuples(self):
        pattern = list()
        for gi in self.gradual_items:
            temp = tuple([gi.attribute_col, gi.symbol])
            pattern.append(temp)
        return pattern

    def get_attributes(self):
        attrs = list()
        syms = list()
        for item in self.gradual_items:
            gi = item.as_integer()
            attrs.append(gi[0])
            syms.append(gi[1])
        return attrs, syms

    def get_index(self, gi):
        for i in range(len(self.gradual_items)):
            gi_obj = self.gradual_items[i]
            if (gi.symbol == gi_obj.symbol) and (gi.attribute_col == gi_obj.attribute_col):
                return i
        return -1

    def inv_pattern(self):
        pattern = list()
        for gi in self.gradual_items:
            pattern.append(gi.inv().tolist())
        return pattern

    def contains(self, gi):
        if gi is None:
            return False
        if gi in self.gradual_items:
            return True
        return False

    def contains_attr(self, gi):
        if gi is None:
            return False
        for gi_obj in self.gradual_items:
            if gi.attribute_col == gi_obj.attribute_col:
                return True
        return False

    def to_string(self):
        pattern = list()
        for item in self.gradual_items:
            pattern.append(item.to_string())
        return pattern

    def to_dict(self):
        gi_dict = {}
        for gi in self.gradual_items:
            gi_dict.update({gi.as_string(): 0})
        return gi_dict

    def print(self, columns):
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


# -------- GRADUAL PATTERNS (START)-------------


# -------- ANT COLONY OPTIMIZATION (START)-------------

"""
CHANGES:
1. generates distance matrix (d_matrix)
2. uses plain methods
"""


def generate_d(valid_bins):
    v_bins = valid_bins
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
    return d, attr_keys


def run_ant_colony(f_path, min_supp=MIN_SUPPORT, evaporation_factor=EVAPORATION_FACTOR,
                   max_iteration=MAX_ITERATIONS):
    # 0. Initialize and prepare data set
    d_set = Dataset(f_path, min_supp)
    d_set.init_gp_attributes()
    # attr_index = d_set.attr_cols
    # e_factor = evaporation_factor
    d, attr_keys = generate_d(d_set.valid_bins)  # distance matrix (d) & attributes corresponding to d

    a = d_set.attr_size
    winner_gps = list()  # subsets
    loser_gps = list()  # supersets
    str_winner_gps = list()  # subsets
    repeated = 0
    it_count = 0
    counter = 0
    max_it = max_iteration

    if d_set.no_bins:
        return []

    # 1. Remove d[i][j] < frequency-count of min_supp
    fr_count = ((min_supp * a * (a - 1)) / 2)
    d[d < fr_count] = 0

    # 3. Initialize pheromones (p_matrix)
    pheromones = np.ones(d.shape, dtype=float)

    # 4. Iterations for ACO
    # while repeated < 1:
    while counter < max_it:
        rand_gp, pheromones = generate_aco_gp(attr_keys, d, pheromones, evaporation_factor)
        if len(rand_gp.gradual_items) > 1:
            # print(rand_gp.get_pattern())
            exits = is_duplicate(rand_gp, winner_gps, loser_gps)
            if not exits:
                repeated = 0
                # check for anti-monotony
                is_super = check_anti_monotony(loser_gps, rand_gp, subset=False)
                is_sub = check_anti_monotony(winner_gps, rand_gp, subset=True)
                if is_super or is_sub:
                    continue
                gen_gp = validate_gp(d_set, rand_gp)
                is_present = is_duplicate(gen_gp, winner_gps, loser_gps)
                is_sub = check_anti_monotony(winner_gps, gen_gp, subset=True)
                if is_present or is_sub:
                    repeated += 1
                else:
                    if gen_gp.support >= min_supp:
                        pheromones = update_pheromones(attr_keys, gen_gp, pheromones)
                        winner_gps.append(gen_gp)
                        str_winner_gps.append(gen_gp.print(d_set.titles))
                    else:
                        loser_gps.append(gen_gp)
                if set(gen_gp.get_pattern()) != set(rand_gp.get_pattern()):
                    loser_gps.append(rand_gp)
            else:
                repeated += 1
        it_count += 1
        if max_it == 1:
            counter = repeated
        else:
            counter = it_count
    # Output
    out = {"Best Patterns": str_winner_gps, "Iterations": it_count}
    return out


def generate_aco_gp(attr_keys, d, p_matrix, e_factor):
    v_matrix = d
    pattern = GP()

    # 1. Generate gradual items with highest pheromone and visibility
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
            gi = GI.parse_gi(attr_keys[j])
            if not pattern.contains_attr(gi):
                pattern.add_gradual_item(gi)
        except IndexError:
            continue

    # 2. Evaporate pheromones by factor e
    p_matrix = (1 - e_factor) * p_matrix
    return pattern, p_matrix


def update_pheromones(attr_keys, pattern, p_matrix):
    idx = [attr_keys.index(x.as_string()) for x in pattern.gradual_items]
    for n in range(len(idx)):
        for m in range(n + 1, len(idx)):
            i = idx[n]
            j = idx[m]
            p_matrix[i][j] += 1
            p_matrix[j][i] += 1
    return p_matrix


def validate_gp(d_set, pattern):
    # pattern = [('2', '+'), ('4', '+')]
    min_supp = d_set.thd_supp
    n = d_set.attr_size
    gen_pattern = GP()
    bin_arr = np.array([])

    for gi in pattern.gradual_items:
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
        return pattern
    else:
        return gen_pattern


def check_anti_monotony(lst_p, pattern, subset=True):
    result = False
    if subset:
        for pat in lst_p:
            result1 = set(pattern.get_pattern()).issubset(set(pat.get_pattern()))
            result2 = set(pattern.inv_pattern()).issubset(set(pat.get_pattern()))
            if result1 or result2:
                result = True
                break
    else:
        for pat in lst_p:
            result1 = set(pattern.get_pattern()).issuperset(set(pat.get_pattern()))
            result2 = set(pattern.inv_pattern()).issuperset(set(pat.get_pattern()))
            if result1 or result2:
                result = True
                break
    return result


def is_duplicate(pattern, lst_winners, lst_losers):
    for pat in lst_losers:
        if set(pattern.get_pattern()) == set(pat.get_pattern()) or \
                set(pattern.inv_pattern()) == set(pat.get_pattern()):
            return True
    for pat in lst_winners:
        if set(pattern.get_pattern()) == set(pat.get_pattern()) or \
                set(pattern.inv_pattern()) == set(pat.get_pattern()):
            return True
    return False

# -------- ANT COLONY (END)-------------
