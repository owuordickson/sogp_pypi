# -*- coding: utf-8 -*-
"""
@author: Dickson Owuor
@credits: Thomas Runkler, Edmond Menya, and Anne Laurent
@license: MIT
@email: owuordickson@gmail.com
@created: 21 July 2021
@modified: 27 October 2022

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


import gc
import math
import json
import time
import random
import numpy as np
import skfuzzy as fuzzy
import multiprocessing as mp
from ypstruct import structure
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import mutual_info_regression

from .__configs__ import *
from .data_gp import DataGP
from .gradual_patterns import GI, ExtGP, TGP, TimeDelay


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

    >>> import so4gp as sgp
    >>> import pandas
    >>> dummy_data = [[30, 3, 1, 10], [35, 2, 2, 8], [40, 4, 2, 7], [50, 1, 1, 6], [52, 7, 1, 2]]
    >>> dummy_df = pandas.DataFrame(dummy_data, columns=['Age', 'Salary', 'Cars', 'Expenses'])
    >>>
    >>> mine_obj = sgp.AntGRAANK(dummy_df, 0.5, max_iter=3, e_factor=0.5)
    >>> result_json = mine_obj.discover()
    >>> print(result_json) # doctest: +SKIP
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

        >>> import so4gp as sgp
        >>> import pandas
        >>> dummy_data = [[30, 3, 1, 10], [35, 2, 2, 8], [40, 4, 2, 7], [50, 1, 1, 6], [52, 7, 1, 2]]
        >>> dummy_df = pandas.DataFrame(dummy_data, columns=['Age', 'Salary', 'Cars', 'Expenses'])
        >>>
        >>> mine_obj = sgp.AntGRAANK(dummy_df, 0.5, max_iter=3, e_factor=0.5)
        >>> result_json = mine_obj.discover()
        >>> print(result_json) # doctest: +SKIP
        {"Algorithm": "ACO-GRAANK", "Best Patterns": [[["Expenses-", "Age+"], 1.0]], "Invalid Count": 1, "Iterations":3}

        :param args: [required] data-source, [optional] minimum-support
        :param max_iter: maximum_iteration, default is 1
        :param e_factor: evaporation factor, default is 0.5

        """
        super(AntGRAANK, self).__init__(*args)
        self.evaporation_factor = e_factor
        """:type evaporation_factor: float"""
        self.max_iteration = max_iter
        """type: max_iteration: int"""
        self.distance_matrix = None
        """:type distance_matrix: numpy.ndarray | None"""
        self.attribute_keys = None
        """:type attribute_keys: list | None"""

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

        :param p_matrix: pheromone matrix
        :type p_matrix: np.ndarray
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
        :type pattern: so4gp.ExtGP

        :param p_matrix: an existing pheromone matrix
        :type p_matrix: numpy.ndarray
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

    >>> import so4gp as sgp
    >>> import pandas
    >>> dummy_data = [[30, 3, 1, 10], [35, 2, 2, 8], [40, 4, 2, 7], [50, 1, 1, 6], [52, 7, 1, 2]]
    >>> dummy_df = pandas.DataFrame(dummy_data, columns=['Age', 'Salary', 'Cars', 'Expenses'])
    >>>
    >>> mine_obj = sgp.ClusterGP(dummy_df, 0.5, max_iter=3, e_prob=0.5)
    >>> result_json = mine_obj.discover()
    >>> print(result_json) # doctest: +SKIP
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

        >>> import so4gp as sgp
        >>> import pandas
        >>> dummy_data = [[30, 3, 1, 10], [35, 2, 2, 8], [40, 4, 2, 7], [50, 1, 1, 6], [52, 7, 1, 2]]
        >>> dummy_df = pandas.DataFrame(dummy_data, columns=['Age', 'Salary', 'Cars', 'Expenses'])
        >>>
        >>> mine_obj = sgp.ClusterGP(dummy_df, 0.5, max_iter=3, e_prob=0.5, no_prob=True)
        >>> result_json = mine_obj.discover()
        >>> print(result_json) # doctest: +SKIP

        :param args: [required] data-source, [optional] minimum-support
        :param e_prob: [optional] erasure probability, the default is 0.5
        :param max_iter: [optional] maximum iteration for score vector estimation, the default is 10
        """
        super(ClusterGP, self).__init__(*args)
        self.erasure_probability = e_prob
        """:type erasure_probability: float"""
        self.max_iteration = max_iter
        """:type max_iteration: int"""
        self.gradual_items, self.cum_wins, self.net_win_mat, self.ij = self._construct_matrices(e_prob)
        """:type gradual_items: np.ndarray"""
        """:type cum_wins: np.ndarray"""
        """:type net_win_mat: np.ndarray"""
        """:type ij: np.ndarray"""
        self.win_mat = np.array([])
        """:type win_mat: np.ndarray"""
        if no_prob:
            self.gradual_items, self.win_mat, self.cum_wins, self.net_win_mat, self.ij = self._construct_all_matrices()

    def _construct_matrices(self, e):
        """Description

        Generates all the gradual items and, constructs: (1) net-win matrix, (2) cumulative wins, (3) pairwise objects.

        :param e: [required] erasure probability
        :type e: float

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
        for col in self.attr_cols:
            # Feature data objects
            col_data = np.array(attr_data[col], dtype=float)  # Feature data objects

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
        # nodes_mat = []  # FP nodes matrix

        # 3. Construct S matrix from data set
        for col in self.attr_cols:
            # Feature data objects
            col_data = np.array(attr_data[col], dtype=float)  # Feature data objects

            # Cumulative Wins: for estimation of score-vector
            temp_cum_wins = np.where(col_data[pair_ij[:, 0]] < col_data[pair_ij[:, 1]], 1,
                                     np.where(col_data[pair_ij[:, 0]] > col_data[pair_ij[:, 1]], -1, 0))

            # S-vector
            s_vec = np.zeros((n,), dtype=np.int32)
            # nodes_vec = [[set(), set()]] * n
            for w in [1, -1]:
                positions = np.flatnonzero(temp_cum_wins == w)
                i, counts_i = np.unique(pair_ij[positions, 0], return_counts=True)
                j, counts_j = np.unique(pair_ij[positions, 1], return_counts=True)
                s_vec[i] += w * counts_i  # i wins/loses (1/-1)
                s_vec[j] += -w * counts_j  # j loses/wins (1/-1)

                """
                if w == 1:
                    for node_i in i:
                        nodes_j = j[np.where(j > node_i)]
                        tmp = nodes_vec[node_i][0].union(set(nodes_j))
                        nodes_vec[node_i] = [tmp, nodes_vec[node_i][1]]

                    for node_j in j:
                        nodes_i = i[np.where(i < node_j)]
                        tmp = nodes_vec[node_j][1].union(set(nodes_i))
                        nodes_vec[node_j] = [nodes_vec[node_j][0], tmp]
                elif w == -1:
                    for node_i in i:
                        nodes_j = j[np.where(j > node_i)]
                        tmp = nodes_vec[node_i][1].union(set(nodes_j))
                        nodes_vec[node_i] = [nodes_vec[node_i][0], tmp]

                    for node_j in j:
                        nodes_i = i[np.where(i < node_j)]
                        tmp = nodes_vec[node_j][0].union(set(nodes_i))
                        nodes_vec[node_j] = [tmp, nodes_vec[node_j][1]]

            # print('positions: ' + str(positions) + '; i: ' + str(i) + '; j: ' + str(j) + '; counts: ' + str(counts_i))
            #    print(nodes_vec)
            # print("\n")"""

            # Normalize S-vector
            if np.count_nonzero(s_vec) > 0:
                w_mat.append(np.copy(s_vec))
                # nodes_mat.append(nodes_vec)

                s_vec[s_vec > 0] = 1  # Normalize net wins
                s_vec[s_vec < 0] = -1  # Normalize net loses

                lst_gis.append(GI(col, '+'))
                cum_wins.append(temp_cum_wins)
                s_mat.append(s_vec)

                lst_gis.append(GI(col, '-'))
                cum_wins.append(-temp_cum_wins)
                s_mat.append(-s_vec)

        # print(np.array(nodes_mat))
        return np.array(lst_gis), np.array(w_mat), np.array(cum_wins), np.array(s_mat), pair_ij

    def _infer_gps(self, clusters):
        """Description

        A function that infers GPs from clusters of gradual items.

        :param clusters: [required] groups of gradual items clustered through K-MEANS algorithm
        :type clusters: np.ndarray

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
        :type c_wins: np.ndarray

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
        for _ in range(self.max_iteration):
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
        :type score_vectors: list

        :return: estimated support (float)
        """

        # Estimate support - use different score-vectors to construct pairs
        n = self.row_count
        bin_mat = np.ones((n, n), dtype=bool)
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
        :type testing: bool

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
        str_gps, estimated_gps = self._infer_gps(y_predicted)

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

    >>> import so4gp as sgp
    >>> import pandas
    >>> dummy_data = [[30, 3, 1, 10], [35, 2, 2, 8], [40, 4, 2, 7], [50, 1, 1, 6], [52, 7, 1, 2]]
    >>> dummy_df = pandas.DataFrame(dummy_data, columns=['Age', 'Salary', 'Cars', 'Expenses'])
    >>>
    >>> mine_obj = sgp.GeneticGRAANK(dummy_df, 0.5, max_iter=3, n_pop=10)
    >>> result_json = mine_obj.discover()
    >>> print(result_json) # doctest: +SKIP
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

        >>> import so4gp as sgp
        >>> import pandas
        >>> dummy_data = [[30, 3, 1, 10], [35, 2, 2, 8], [40, 4, 2, 7], [50, 1, 1, 6], [52, 7, 1, 2]]
        >>> dummy_df = pandas.DataFrame(dummy_data, columns=['Age', 'Salary', 'Cars', 'Expenses'])
        >>>
        >>> mine_obj = sgp.GeneticGRAANK(dummy_df, 0.5, max_iter=3, n_pop=10)
        >>> result_json = mine_obj.discover()
        >>> print(result_json) # doctest: +SKIP
        {"Algorithm": "GA-GRAANK", "Best Patterns": [[["Age+", "Salary+", "Expenses-"], 0.6]], "Invalid Count": 12,
        "Iterations": 2}


        :param args: [required] data-source, [optional] minimum-support
        :param max_iter: maximum_iteration, default is 1
        :type max_iter: int

        :param n_pop: initial individual population, default is 5
        :type n_pop: int

        :param pc: children proportion, default is 0.5
        :type pc: float

        :param gamma: cross-over gamma ratio, default is 1
        :type gamma: float

        :param mu: mutation mu ratio, default is 0.9
        :type mu: float

        :param sigma: mutation sigma ratio, default is 0.9
        :type sigma: float
        """
        super(GeneticGRAANK, self).__init__(*args)
        self.max_iteration = max_iter
        """type: max_iteration: int"""
        self.n_pop = n_pop
        """type: n_pop: int"""
        self.pc = pc
        """type: pc: float"""
        self.gamma = gamma
        """type: gamma: float"""
        self.mu = mu
        """type: mu: float"""
        self.sigma = sigma
        """type: sigma: float"""

    def _crossover(self, p1: structure, p2: structure):
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

    def _mutate(self, x: structure):
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

        >>> import so4gp as sgp
        >>> import pandas
        >>> dummy_data = [[30, 3, 1, 10], [35, 2, 2, 8], [40, 4, 2, 7], [50, 1, 1, 6], [52, 7, 1, 2]]
        >>> dummy_df = pandas.DataFrame(dummy_data, columns=['Age', 'Salary', 'Cars', 'Expenses'])
        >>>
        >>> mine_obj = sgp.GRAANK(data_source=dummy_df, min_sup=0.5, eq=False)
        >>> result_json = mine_obj.discover()
        >>> print(result_json) # doctest: +SKIP

        """

    def _gen_apriori_candidates(self, gi_bins: list, target_col: int =None):
        """Description

        Generates Apriori GP candidates (w.r.t target-column/reference-column if available)
        :param gi_bins: GI together with bitmaps
        :return:
        """
        min_sup = self.thd_supp
        n = self.attr_size

        invalid_count = 0
        res = []
        all_candidates = []
        if len(gi_bins) < 2:
            return []

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
                if target_col is None:
                    use_target_col = True
                else:
                    has_tgt_col = np.array([(y[0] == target_col) for y in gp_cand], dtype=bool)
                    use_target_col = np.any(has_tgt_col)
                if (use_target_col and
                        (len(gp_cand) == len(gi_o) + 1) and
                        (not (all_candidates != [] and gp_cand in all_candidates)) and
                        (not (all_candidates != [] and inv_gp_cand in all_candidates))):
                    test = 1
                    repeated_attr = -1
                    for k in gp_cand:
                        if k[0] == repeated_attr:
                            test = 0
                            break
                        else:
                            repeated_attr = k[0]
                    if test == 1:
                        m = gi_bins[i][1] * gi_bins[j][1]
                        sup = float(np.sum(m)) / float(n * (n - 1.0) / 2.0)
                        if sup > min_sup:
                            res.append([gp_cand, m, sup])
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
        valid_bins = self.valid_bins

        invalid_count = 0
        while len(valid_bins) > 0:
            valid_bins, inv_count = self._gen_apriori_candidates(valid_bins)
            invalid_count += inv_count
            for v_bin in valid_bins:
                gi_arr = v_bin[0]
                # bin_data = v_bin[1]
                sup = v_bin[2]
                self.gradual_patterns = ExtGP.remove_subsets(self.gradual_patterns, set(gi_arr))

                gp = ExtGP()
                """:type gp: ExtGP"""
                for obj in gi_arr:
                    gi = GI(obj[0], obj[1].decode())
                    """:type gi: GI"""
                    gp.add_gradual_item(gi)
                gp.set_support(sup)
                self.gradual_patterns.append(gp)
                str_winner_gps.append(gp.print(self.titles))
        # Output
        out = json.dumps({"Algorithm": "GRAANK", "Patterns": str_winner_gps, "Invalid Count": invalid_count})
        """:type out: object"""
        return out

    @staticmethod
    def decompose_to_gp_component(pairwise_objs: np.ndarray):
        """
        A method that decomposes the pairwise matrix of a gradual item/pattern into a warping path. This path is the
        decomposed component of that gradual item/pattern.

        :param pairwise_objs:
        :return: dict of GP components
        """
        gp_components = {}
        """:type: dict"""

        for pairwise_obj in pairwise_objs:
            pairwise_mat = pairwise_obj[1]
            attr_col = pairwise_obj[0][0]
            attr_name = pairwise_obj[0][1].decode()
            gi = GI(attr_col, attr_name)
            edge_lst = [(i, j) for i, row in enumerate(pairwise_mat) for j, val in enumerate(row) if val]
            gp_components[gi.to_string()] = edge_lst
        return gp_components


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

    >>> import so4gp as sgp
    >>> import pandas
    >>> dummy_data = [[30, 3, 1, 10], [35, 2, 2, 8], [40, 4, 2, 7], [50, 1, 1, 6], [52, 7, 1, 2]]
    >>> dummy_df = pandas.DataFrame(dummy_data, columns=['Age', 'Salary', 'Cars', 'Expenses'])
    >>>
    >>> mine_obj = sgp.HillClimbingGRAANK(dummy_df, 0.5, max_iter=3, step_size=0.5)
    >>> result_json = mine_obj.discover()
    >>> print(result_json) # doctest: +SKIP

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

        >>> import so4gp as sgp
        >>> import pandas
        >>> dummy_data = [[30, 3, 1, 10], [35, 2, 2, 8], [40, 4, 2, 7], [50, 1, 1, 6], [52, 7, 1, 2]]
        >>> dummy_df = pandas.DataFrame(dummy_data, columns=['Age', 'Salary', 'Cars', 'Expenses'])
        >>>
        >>> mine_obj = sgp.HillClimbingGRAANK(dummy_df, 0.5, max_iter=3, step_size=0.5)
        >>> result_json = mine_obj.discover()
        >>> print(result_json) # doctest: +SKIP
        {"Algorithm": "LS-GRAANK", "Best Patterns": [[["Age+", "Expenses-"], 1.0]], "Invalid Count": 2, "Iterations": 2}

        :param args: [required] data-source, [optional] minimum-support
        :param max_iter: maximum_iteration, default is 1
        :param step_size: step size, default is 0.5
        """
        super(HillClimbingGRAANK, self).__init__(*args)
        self.step_size = step_size
        """type: step_size: int"""
        self.max_iteration = max_iter
        """type: max_iteration: int"""

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
    def decode_gp(attr_keys: list, position: float):
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
    def cost_function(position: float, attr_keys: list, d_set: DataGP):
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
    def apply_bound(x: structure, var_min: int, var_max: int):
        """Description

        Modifies x (a numeric value) if it exceeds the lower/upper bound of the numeric search space.

        :param x: a value in the numeric search space
        :param var_min: lower-bound value
        :param var_max: upper-bound value
        :return: nothing
        """

        x.position = np.maximum(x.position, var_min)
        x.position = np.minimum(x.position, var_max)


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

    >>> import so4gp as sgp
    >>> import pandas
    >>> dummy_data = [[30, 3, 1, 10], [35, 2, 2, 8], [40, 4, 2, 7], [50, 1, 1, 6], [52, 7, 1, 2]]
    >>> dummy_df = pandas.DataFrame(dummy_data, columns=['Age', 'Salary', 'Cars', 'Expenses'])
    >>>
    >>> mine_obj = sgp.ParticleGRAANK(dummy_df, 0.5, max_iter=3, n_particle=10)
    >>> result_json = mine_obj.discover()
    >>> print(result_json) # doctest: +SKIP
    {"Algorithm": "PSO-GRAANK", "Best Patterns": [], "Invalid Count": 12, "Iterations": 2}


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

        >>> import so4gp as sgp
        >>> import pandas
        >>> dummy_data = [[30, 3, 1, 10], [35, 2, 2, 8], [40, 4, 2, 7], [50, 1, 1, 6], [52, 7, 1, 2]]
        >>> dummy_df = pandas.DataFrame(dummy_data, columns=['Age', 'Salary', 'Cars', 'Expenses'])
        >>>
        >>> mine_obj = sgp.ParticleGRAANK(dummy_df, 0.5, max_iter=3, n_particle=10)
        >>> result_json = mine_obj.discover()
        >>> print(result_json) # doctest: +SKIP
        {"Algorithm": "PSO-GRAANK", "Best Patterns": [], "Invalid Count": 12, "Iterations": 2}

        :param args: [required] data-source, [optional] minimum-support
        :param max_iter: maximum_iteration, default is 1
        :param n_particle: initial particle population, default is 5
        :param vel: velocity, default is 0.9
        :param coeff_p: personal coefficient, default is 0.01
        :param coeff_g: global coefficient, default is 0.9
        """
        super(ParticleGRAANK, self).__init__(*args)
        self.max_iteration = max_iter
        """type: max_iteration: int"""
        self.n_particles = n_particle
        """type: n_particles: int"""
        self.velocity = vel
        """type: velocity: float"""
        self.coeff_p = coeff_p
        """type: coeff_p: float"""
        self.coeff_g = coeff_g
        """type: coeff_g: float"""

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

    >>> import so4gp as sgp
    >>> import pandas
    >>> dummy_data = [[30, 3, 1, 10], [35, 2, 2, 8], [40, 4, 2, 7], [50, 1, 1, 6], [52, 7, 1, 2]]
    >>> dummy_df = pandas.DataFrame(dummy_data, columns=['Age', 'Salary', 'Cars', 'Expenses'])
    >>>
    >>> mine_obj = sgp.RandomGRAANK(dummy_df, 0.5, max_iter=3)
    >>> result_json = mine_obj.discover()
    >>> print(result_json) # doctest: +SKIP

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

        >>> import so4gp as sgp
        >>> import pandas
        >>> dummy_data = [[30, 3, 1, 10], [35, 2, 2, 8], [40, 4, 2, 7], [50, 1, 1, 6], [52, 7, 1, 2]]
        >>> dummy_df = pandas.DataFrame(dummy_data, columns=['Age', 'Salary', 'Cars', 'Expenses'])
        >>>
        >>> mine_obj = sgp.RandomGRAANK(dummy_df, 0.5, max_iter=3)
        >>> result_json = mine_obj.discover()
        >>> print(result_json) # doctest: +SKIP
        {"Algorithm": "RS-GRAANK", "Best Patterns": [[["Age+", "Salary+", "Expenses-"], 0.6]], "Invalid Count": 1,
        "Iterations": 3}

        :param args: [required] data-source, [optional] minimum-support
        :param max_iter: maximum_iteration, default is 1
        """
        super(RandomGRAANK, self).__init__(*args)
        self.max_iteration = max_iter
        """type: max_iteration: int"""

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


class TGrad(GRAANK):
    """Description of class TGrad.

    TGrad is an algorithm that is used to extract temporal gradual patterns from numeric datasets. An algorithm for
    mining temporal gradual patterns using fuzzy membership functions. It uses technique published
    in: https://ieeexplore.ieee.org/abstract/document/8858883.

    """

    def __init__(self, f_path: str, eq: bool, min_sup: float, target_col: int, min_rep: float, num_cores: int):
        """
        TGrad is an algorithm that is used to extract temporal gradual patterns from numeric datasets.

        :param f_path: path to ddtaset file
        :param eq: are equal object considered in GP matrix.
        :param min_sup: minimum support value.
        :param target_col: Target column.
        :param min_rep: minimum representativity value.
        :param num_cores: number of cores to use.

        >>> import so4gp as sgp
        >>> import pandas
        >>> dummy_data = [["2021-03", 30, 3, 1, 10], ["2021-03", 35, 2, 2, 8], ["2021-03", 40, 4, 2, 7], ["2021-03", 50, 1, 1, 6], ["2021-03", 52, 7, 1, 2]]
        >>> dummy_df = pandas.DataFrame(dummy_data, columns=['Date', 'Age', 'Salary', 'Cars', 'Expenses'])
        >>>
        >>> mine_obj = sgp.TGrad(dummy_df, min_sup=0.5, target_col=1, min_rep=0.5)
        >>> result_json = mine_obj.discover_tgp(parallel=True)
        >>> print(result_json)
        """

        super(TGrad, self).__init__(data_source=f_path, min_sup=min_sup, eq=eq)
        self.target_col = target_col
        """:type: target_col: int"""
        self.min_rep = min_rep
        """:type: min_rep: float"""
        self.max_step = self.row_count - int(min_rep * self.row_count)
        """:type: max_step: int"""
        self.full_attr_data = self.data.copy().T
        """:type: full_attr_data: numpy.ndarray"""
        self.cores = num_cores
        """:type: cores int"""
        if len(self.time_cols) > 0:
            print("Dataset Ok")
            self.time_ok = True
            """:type: time_ok: bool"""
        else:
            print("Dataset Error")
            self.time_ok = False
            """:type: time_ok: bool"""
            raise Exception('No date-time datasets found')

    def discover_tgp(self, parallel: bool = False):
        """

        Applies fuzzy-logic, data transformation and gradual pattern mining to mine for Fuzzy Temporal Gradual Patterns.

        :param parallel: allow multi-processing.
        :return: list
        """

        if parallel:
            # implement parallel multi-processing
            steps = range(self.max_step)
            pool = mp.Pool(self.cores)
            patterns = pool.map(self.transform_and_mine, steps)
            pool.close()
            pool.join()
            return patterns
        else:
            patterns = list()
            for step in range(self.max_step):
                t_gps = self.transform_and_mine(step + 1)  # because for-loop is not inclusive from range: 0 - max_step
                if t_gps:
                    patterns.append(t_gps)
            return patterns

    def transform_and_mine(self, step: int, return_patterns: bool = True):
        """
        A method that: (1) transforms data according to a step value and, (2) mines the transformed data for FTGPs.

        :param step: data transformation step.
        :param return_patterns: allow method to mine TGPs.
        :return: list of TGPs
        """
        # NB: Restructure dataset based on target/reference col
        if self.time_ok:
            # 1. Calculate time difference using step
            ok, time_diffs = self.get_time_diffs(step)
            if not ok:
                msg = "Error: Time in row " + str(time_diffs[0]) \
                      + " or row " + str(time_diffs[1]) + " is not valid."
                raise Exception(msg)
            else:
                tgt_col = self.target_col
                if tgt_col in self.time_cols:
                    msg = "Target column is a 'date-time' attribute"
                    raise Exception(msg)
                elif (tgt_col < 0) or (tgt_col >= self.col_count):
                    msg = "Target column does not exist\nselect column between: " \
                          "0 and " + str(self.col_count - 1)
                    raise Exception(msg)
                else:
                    # 2. Transform datasets
                    delayed_attr_data = None
                    n = self.row_count
                    for col_index in range(self.col_count):
                        # Transform the datasets using (row) n+step
                        if (col_index == tgt_col) or (col_index in self.time_cols):
                            # date-time column OR target column
                            temp_row = self.full_attr_data[col_index][0: (n - step)]
                        else:
                            # other attributes
                            temp_row = self.full_attr_data[col_index][step: n]

                        delayed_attr_data = temp_row if (delayed_attr_data is None) \
                            else np.vstack((delayed_attr_data, temp_row))
                    # print(f"Time Diffs: {time_diffs}\n")
                    # print(f"{self.full_attr_data}: {type(self.full_attr_data)}\n")
                    # print(f"{delayed_attr_data}: {type(delayed_attr_data)}\n")

                    if return_patterns:
                        # 2. Execute t-graank for each transformation
                        t_gps = self.discover(t_diffs=time_diffs, attr_data=delayed_attr_data)
                        if len(t_gps) > 0:
                            return t_gps
                        return False
                    else:
                        return delayed_attr_data, time_diffs
        else:
            msg = "Fatal Error: Time format in column could not be processed"
            raise Exception(msg)

    def get_time_diffs(self, step: int):  # optimized
        """

        A method that computes the difference between 2 timestamps separated by a specific transformation step.

        :param step: data transformation step.
        :return: dict of time delay values
        """
        size = self.row_count
        time_diffs = {}  # {row: time-lag}
        for i in range(size):
            if i < (size - step):
                stamp_1 = 0
                stamp_2 = 0
                for col in self.time_cols:  # sum timestamps from all time-columns
                    temp_1 = str(self.data[i][int(col)])
                    temp_2 = str(self.data[i + step][int(col)])
                    temp_stamp_1 = TGrad.get_timestamp(temp_1)
                    temp_stamp_2 = TGrad.get_timestamp(temp_2)
                    if (not temp_stamp_1) or (not temp_stamp_2):
                        # Unable to read time
                        return False, [i + 1, i + step + 1]
                    else:
                        stamp_1 += temp_stamp_1
                        stamp_2 += temp_stamp_2
                time_diff = (stamp_2 - stamp_1)
                # if time_diff < 0:
                # Error time CANNOT go backwards
                # print(f"Problem {i} and {i + step} - {self.time_cols}")
                #    return False, [i + 1, i + step + 1]
                time_diffs[int(i)] = float(abs(time_diff))
        return True, time_diffs

    def discover(self, t_diffs: np.ndarray | dict = None, attr_data: np.ndarray = None):
        """

        Uses apriori algorithm to find GP candidates based on the target-attribute. The candidates are validated if
        their computed support is greater than or equal to the minimum support threshold specified by the user.

        :param t_diffs: time-delay values
        :param attr_data: the transformed data.
        :return: temporal-GPs as a list.
        """

        self.fit_bitmap(attr_data)

        gradual_patterns = []
        """:type gradual_patterns: list"""
        valid_bins = self.valid_bins

        invalid_count = 0
        while len(valid_bins) > 0:
            valid_bins, inv_count = self._gen_apriori_candidates(valid_bins, self.target_col)
            invalid_count += inv_count
            for v_bin in valid_bins:
                gi_arr = v_bin[0]
                bin_data = v_bin[1]
                sup = v_bin[2]
                gradual_patterns = TGP.remove_subsets(gradual_patterns, set(gi_arr))
                t_lag = self.get_fuzzy_time_lag(bin_data, t_diffs, gi_arr)

                if t_lag.valid:
                    tgp = TGP()
                    """:type gp: TGP"""
                    for obj in gi_arr:
                        gi = GI(obj[0], obj[1].decode())
                        """:type gi: GI"""
                        if gi.attribute_col == self.target_col:
                            tgp.add_target_gradual_item(gi)
                        else:
                            tgp.add_temporal_gradual_item(gi, t_lag)
                    tgp.set_support(sup)
                    gradual_patterns.append(tgp)
        return gradual_patterns

    def get_fuzzy_time_lag(self, bin_data: np.ndarray, time_diffs: np.ndarray | dict, gi_arr: set = None, use_clustering_method: bool = False):
        """

        A method that uses fuzzy membership function to select the most accurate time-delay value. We implement two
        methods: (1) uses classical slide and re-calculate dynamic programming to find best time-delay value and,
        (2) uses metaheuristic hill-climbing to find the best time-delay value.

        :param bin_data: gradual item pairwise matrix.
        :param time_diffs: time-delay values.
        :param gi_arr: gradual item object.
        :param use_clustering_method: find and approximate best time-delay value using KMeans and Hill-climbing approach.
        :return: TimeDelay object.
        """

        # 1. Get Indices
        indices = np.argwhere(bin_data == 1)

        # 2. Get TimeDelay Array
        selected_rows = np.unique(indices.flatten())
        if isinstance(self, TGradAMI):
            selected_cols = []
            for obj in gi_arr:
                # Ignore target-col and, remove time-cols and target-col from count
                col = int(obj[0])
                if (col != self.target_col) and (col < self.target_col):
                    selected_cols.append(col - (len(self.time_cols)))
                elif (col != self.target_col) and (col > self.target_col):
                    selected_cols.append(col - (len(self.time_cols) + 1))
            selected_cols = np.array(selected_cols, dtype=int)
            t_lag_arr = time_diffs[np.ix_(selected_cols, selected_rows)]
        else:
            time_lags = []
            for row, stamp_diff in time_diffs.items():  # {row: time-lag-stamp}
                if int(row) in selected_rows:
                    time_lags.append(stamp_diff)
            t_lag_arr = np.array(time_lags)
            best_time_lag = TGrad.approx_time_slide_calculate(t_lag_arr)
            return best_time_lag

        # 3. Approximate TimeDelay value
        best_time_lag = TimeDelay(-1, 0)
        """:type best_time_lag: so4gp.TimeDelay"""
        if use_clustering_method:
            # 3b. Learn the best MF through slide-descent/sliding
            a, b, c = self.tri_mf_data
            best_time_lag = TimeDelay(-1, -1)
            fuzzy_set = []
            for t_lags in t_lag_arr:
                init_bias = abs(b - np.median(t_lags))
                slide_val, loss = TGradAMI.approx_time_hill_climbing(a, b, c, t_lags, initial_bias=init_bias)
                tstamp = int(b - slide_val)
                sup = float(1 - loss)
                fuzzy_set.append([tstamp, float(loss)])
                if sup >= best_time_lag.support and abs(tstamp) > abs(best_time_lag.timestamp):
                    best_time_lag = TimeDelay(tstamp, sup)
                # print(f"New Membership Fxn: {a - slide_val}, {b - slide_val}, {c - slide_val}")
        else:
            # 3a. Approximate TimeDelay using Fuzzy Membership
            for t_lags in t_lag_arr:
                time_lag = TGrad.approx_time_slide_calculate(t_lags)
                if time_lag.support >= best_time_lag.support:
                    best_time_lag = time_lag
        return best_time_lag

    @staticmethod
    def get_timestamp(time_str: str):
        """

        A method that computes the corresponding timestamp from a DateTime string.

        :param time_str: DateTime value as a string
        :return: timestamp value
        """
        try:
            ok, stamp = DataGP.test_time(time_str)
            if ok:
                return stamp
            else:
                return False
        except ValueError:
            return False

    @staticmethod
    def triangular_mf(x: float, a: float, b: float, c: float):
        """

        A method that implements the fuzzy triangular membership function and computes the membership degree of value w.r.t
        the MF.

        :param x: value to be tested.
        :param a: left-side/minimum boundary of the triangular membership function.
        :param b: center value of the triangular membership function.
        :param c: maximum boundary value of the triangular membership function.
        :return: membership degree of value x.
        """
        if a <= x <= b:
            return (x - a) / (b - a)
        elif b <= x <= c:
            return (c - x) / (c - b)
        else:
            return 0

    @staticmethod
    def approx_time_slide_calculate(time_lags: np.ndarray):
        """

        A method that selects the most appropriate time-delay value from a list of possible values.

        :param time_lags: an array of all the possible time-delay values.
        :return: the approximated TimeDelay object.
        """

        if len(time_lags) <= 0:
            # if time_lags is blank return nothing
            return TimeDelay()
        else:
            time_lags = np.absolute(np.array(time_lags))
            min_a = np.min(time_lags)
            max_c = np.max(time_lags)
            count = time_lags.size + 3
            tot_boundaries = np.linspace(min_a / 2, max_c + 1, num=count)

            sup1 = 0
            center = time_lags[0]
            size = len(tot_boundaries)
            for i in range(0, size, 2):
                if (i + 3) <= size:
                    boundaries = tot_boundaries[i:i + 3:1]
                else:
                    boundaries = tot_boundaries[size - 3:size:1]
                memberships = fuzzy.membership.trimf(time_lags, boundaries)

                # Compute Support
                sup_count = np.count_nonzero(memberships > 0)
                total = memberships.size
                sup = sup_count / total
                # sup = calculate_support(memberships)

                if sup > sup1:
                    sup1 = sup
                    center = boundaries[1]
                if sup >= 0.5:
                    # print(boundaries[1])
                    return TimeDelay(int(boundaries[1]), sup)
            return TimeDelay(center, sup1)


class TGradAMI(TGrad):

    def __init__(self, f_path: str, eq: bool, min_sup: float, target_col: int, min_rep: float, num_cores: int):
        """"""
        # Compute MI w.r.t. target-column with original dataset to get the actual relationship
        # between variables. Compute MI for every time-delay/time-lag: if the values are
        # almost equal to actual, then we have the most accurate time-delay. Instead of
        # min-representativity value, we propose error-margin.

        super(TGradAMI, self).__init__(f_path, eq, min_sup, target_col=target_col, min_rep=min_rep, num_cores=num_cores)
        # self.error_margin = err
        self.min_membership = 0.001
        self.tri_mf_data = None  # The a,b,c values of the triangular membership function in indices 0,1,2 respectively.
        self.feature_cols = np.setdiff1d(self.attr_cols, self.target_col)
        self.initial_mutual_info = None
        self.mi_arr = None

    def compute_mutual_info(self):
        """"""
        # 1. Compute all the MI for every time-delay and store in list
        mi_list = []
        for step in range(self.max_step):
            attr_data, _ = self.transform_and_mine(step, return_patterns=False)
            y = np.array(attr_data[self.target_col], dtype=float).T
            x_data = np.array(attr_data[self.feature_cols], dtype=float).T
            mutual_info = mutual_info_regression(x_data, y)
            mi_list.append(mutual_info)
        mi_arr = np.array(mi_list, dtype=float)

        # 2. Standardize MI array
        # We replace 0 with -1 because 0 indicates NO MI, so we make it useless by making it -1, so it allows small
        # MI values to be considered and not 0. This is beautiful because if initial MI is 0, then both will be -1
        # making it the optimal MI with any other -1 in the time-delayed MIs
        mi_arr[mi_arr == 0] = -1
        # print(f"{mi_arr}\n")
        self.initial_mutual_info = mi_arr[0]  # step 0 is the MI without any time delay (or step)
        self.mi_arr = mi_arr[1:]

    def gather_delayed_data(self, optimal_dict: dict, max_step: int):
        """"""
        delayed_data = None
        time_data = []
        # time_dict = {}
        n = self.row_count
        k = (n - max_step)  # No. of rows created by largest step-delay
        for col_index in range(self.col_count):
            if (col_index == self.target_col) or (col_index in self.time_cols):
                # date-time column OR target column
                temp_row = self.full_attr_data[col_index][0: k]
            else:
                # other attributes
                step = optimal_dict[col_index]
                temp_row = self.full_attr_data[col_index][step: n]
                _, time_diffs = self.get_time_diffs(step)

                # Get first k items for delayed data
                temp_row = temp_row[0: k]

                # Get first k items for time-lag data
                temp_diffs = [(time_diffs[i]) for i in range(k)]
                time_data.append(temp_diffs)

                # for i in range(k):
                #    if i in time_dict:
                #        time_dict[i].append(time_diffs[i])
                #    else:
                #        time_dict[i] = [time_diffs[i]]
                # print(f"{time_diffs}\n")
                # WHAT ABOUT TIME DIFFERENCE/DELAY? It is different for every step!!!
            delayed_data = temp_row if (delayed_data is None) \
                else np.vstack((delayed_data, temp_row))

        # print(f"{time_dict}\n")
        return delayed_data, np.array(time_data)

    def discover_tgp(self, parallel=False, eval_mode=False):
        """"""

        # 1. Compute mutual information
        self.compute_mutual_info()

        # 2. Identify steps (for every feature w.r.t. target) with minimum error from initial MI
        squared_diff = np.square(np.subtract(self.mi_arr, self.initial_mutual_info))
        absolute_error = np.sqrt(squared_diff)
        optimal_steps_arr = np.argmin(absolute_error, axis=0)
        max_step = (np.max(optimal_steps_arr) + 1)
        # print(f"Largest step delay: {max_step}\n")
        # print(f"Abs.E.: {absolute_error}\n")

        # 3. Integrate feature indices with the computed steps
        # optimal_dict = dict(map(lambda key, val: (int(key), int(val+1)), self.feature_cols, optimal_steps_arr))
        optimal_dict = {int(self.feature_cols[i]): int(optimal_steps_arr[i] + 1) for i in range(len(self.feature_cols))}
        # print(f"Optimal Dict: {optimal_dict}\n")  # {col: steps}

        # 4. Create final (and dynamic) delayed dataset
        delayed_data, time_data = self.gather_delayed_data(optimal_dict, max_step)
        # print(f"{delayed_data}\n")
        # print(f"Time Lags: {time_data}\n")

        # 5. Build triangular MF
        a, b, c = TGradAMI.build_mf_w_clusters(time_data)
        self.tri_mf_data = np.array([a, b, c])
        # print(f"Membership Function: {a}, {b}, {c}\n")

        # 6. Discover temporal-GPs from time-delayed data
        # 6a. Learn the best MF through slide-descent/sliding
        # 6b. Apply cartesian product on multiple MFs to pick the MF with the best center (inference logic)
        # Mine tGPs and then compute Union of time-lag MFs,
        # from this union select the MF with more members (little loss)
        t_gps = self.discover(t_diffs=time_data, attr_data=delayed_data)

        if len(t_gps) > 0:
            if eval_mode:
                title_row = []
                time_title = []
                eval_data = []
                # print(eval_data)
                for txt in self.titles:
                    col = int(txt[0])
                    title_row.append(str(txt[1].decode()))
                    if (col != self.target_col) and (col not in self.time_cols):
                        time_title.append(str(txt[1].decode()))

                return t_gps, np.vstack((np.array(title_row), delayed_data.T)), np.vstack(
                    (np.array(time_title), time_data.T)), eval_data
            else:
                return t_gps
        return False

    @staticmethod
    def build_mf_w_clusters(time_data: np.ndarray):
        """"""

        # Reshape into 1-column dataset
        time_data = time_data.reshape(-1, 1)

        # Standardize data
        scaler = MinMaxScaler()
        data_scaled = scaler.fit_transform(time_data)

        # Apply SVD
        u, s, vt = np.linalg.svd(data_scaled, full_matrices=False)

        # Plot singular values to help determine the number of clusters
        # Based on the plot, choose the number of clusters (e.g., 3 clusters)
        num_clusters = int(s[0])

        # Perform k-means clustering
        kmeans = KMeans(n_clusters=num_clusters)
        kmeans.fit(data_scaled)

        # Get cluster centers
        centers = kmeans.cluster_centers_.flatten()

        # Define membership functions to ensure membership > 0.5
        # mf_list = []
        largest_mf = [0, 0, 0]
        for center in centers:
            half_width = 0.5 / 2  # since membership value should be > 0.5
            a = center - half_width
            b = center
            c = center + half_width
            if abs(c - a) > abs(largest_mf[2] - largest_mf[0]):
                largest_mf = [a, b, c]
            # mf_list.append((a, b, c))

        # Reverse the scaling
        a = scaler.inverse_transform([[largest_mf[0]]])[0, 0]
        b = scaler.inverse_transform([[largest_mf[1]]])[0, 0]
        c = scaler.inverse_transform([[largest_mf[2]]])[0, 0]

        # Shift to remove negative MF (we do not want negative timestamps)
        if a < 0:
            shift_by = abs(a)
            a = a + shift_by
            b = b + shift_by
            c = c + shift_by
        return a, b, c

    @staticmethod
    def approx_time_hill_climbing(a: float, b: float, c: float, x_train: np.ndarray,
                                  initial_bias: float = 0, step_size: float = 0.9, max_iterations: int = 10):
        """"""

        # Normalize x_train
        x_train = np.array(x_train, dtype=float)
        # print(f"x-train: {x_train}")

        # Perform hill climbing to find the optimal bias
        min_membership = 0.001
        bias = initial_bias
        y_train = x_train + bias
        tri_mf = np.array([a, b, c])
        best_mse = TGradAMI.hill_climbing_cost_function(y_train, tri_mf, min_membership)
        for iteration in range(max_iterations):
            # Generate a new candidate bias by perturbing the current bias
            new_bias = bias + step_size * np.random.randn()

            # Compute the predictions and the MSE with the new bias
            y_train = x_train + new_bias
            new_mse = TGradAMI.hill_climbing_cost_function(y_train, tri_mf, min_membership)

            # If the new MSE is lower, update the bias
            if new_mse < best_mse:
                # print(f"new bias: {new_bias}")
                bias = new_bias
                best_mse = new_mse

        # Make predictions using the optimal bias
        # y_train = x_train + bias
        # print(f"Optimal bias: {bias}")
        # print(f"Predictions: {y_train}")
        # print(f"Mean Squared Error: {best_mse*100}%")
        return bias, best_mse

    @staticmethod
    def hill_climbing_cost_function(y_train: np.ndarray, tri_mf: np.ndarray, min_membership: float = 0.5):
        """
        Computes the logistic regression cost function for a fuzzy set created from a
        triangular membership function.

        :param y_train: A numpy array of the predicted labels.
        :param tri_mf: The a,b,c values of the triangular membership function in indices 0,1,2 respectively.
        :param min_membership: The minimum accepted value to allow membership in a fuzzy set.
        :return: cost function values.
        """

        a, b, c = tri_mf[0], tri_mf[1], tri_mf[2]

        # 1. Generate fuzzy data set using MF from x_data
        y_hat = np.where(y_train <= b,
                         (y_train - a) / (b - a),
                         (c - y_train) / (c - b))
        # 2. Generate y_train based on the given criteria (x>minimum_membership)
        y_hat = np.where(y_hat >= min_membership, 1, 0)

        # 3. Compute loss
        hat_count = np.count_nonzero(y_hat)
        true_count = len(y_hat)
        loss = (((true_count - hat_count) / true_count) ** 2) ** 0.5
        # loss = abs(true_count - hat_count)
        return loss
