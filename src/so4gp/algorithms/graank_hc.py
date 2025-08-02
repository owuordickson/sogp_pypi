# -*- coding: utf-8 -*-
# SPDX-License-Identifier: GNU GPL v3
# This file is licensed under the terms of the GNU GPL v3.0.
# See the LICENSE file at the root of this
# repository for complete details.


import json
import random
import numpy as np
from ypstruct import structure

try:
    from ..data_gp import DataGP
    from ..gradual_patterns import GI
    from .numeric_ss import NumericSS
except ImportError:
    from src.so4gp import DataGP, GI
    from src.so4gp.algorithms import NumericSS


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

    """

    def __init__(self, *args, max_iter: int = 1, step_size: float = 0.5, **kwargs):
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

        :param args: [required] data source path of Pandas DataFrame, [optional] minimum-support, [optional] eq
        :param max_iter: [optional] maximum_iteration, default is 1
        :param step_size: [optional] step size, default is 0.5

        >>> import so4gp as sgp
        >>> import pandas
        >>> import json
        >>> dummy_data = [[30, 3, 1, 10], [35, 2, 2, 8], [40, 4, 2, 7], [50, 1, 1, 6], [52, 7, 1, 2]]
        >>> dummy_df = pandas.DataFrame(dummy_data, columns=['Age', 'Salary', 'Cars', 'Expenses'])
        >>>
        >>> mine_obj = sgp.HillClimbingGRAANK(dummy_df, 0.5, max_iter=3, step_size=0.5)
        >>> result_json = mine_obj.discover()
        >>> result = json.loads(result_json)
        >>> # print(result['Patterns'])
        >>> print(result_json) # doctest: +SKIP
        {"Algorithm": "LS-GRAANK", "Best Patterns": [[["Age+", "Expenses-"], 1.0]], "Invalid Count": 2, "Iterations": 2}
        """
        super(HillClimbingGRAANK, self).__init__(*args, **kwargs)
        self.step_size = step_size
        """type: step_size: float"""
        self.max_iteration = max_iter
        """type: max_iteration: int"""
        self.n_var = 1
        """:type n_var: int"""

    def discover(self):
        """Description

        Uses hill-climbing algorithm to find GP candidates. The candidates are validated if their computed support is
        greater than or equal to the minimum support threshold specified by the user.

        :return: JSON object
        """
        # Prepare data set
        self.fit_bitmap()
        attr_keys = [GI(x[0], x[1].decode()).as_string for x in self.valid_bins[:, 0]]

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
            best_sol.position = np.random.uniform(var_min, var_max, self.n_var)
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
