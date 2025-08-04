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

class GeneticGRAANK(DataGP):

    def __init__(self, *args, max_iter=1, n_pop=5, pc=0.5, gamma=1.0, mu=0.9, sigma=0.9, **kwargs):
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

        :param args: [required] data source path of Pandas DataFrame, [optional] minimum-support, [optional] eq
        :param max_iter: [optional] maximum_iteration, default is 1
        :type max_iter: int

        :param n_pop: [optional] initial individual population, default is 5
        :type n_pop: int

        :param pc: [optional] children proportion, default is 0.5
        :type pc: float

        :param gamma: [optional] cross-over gamma ratio, default is 1
        :type gamma: float

        :param mu: [optional] mutation mu ratio, default is 0.9
        :type mu: float

        :param sigma: [optional] mutation sigma ratio, default is 0.9
        :type sigma: float

        >>> import so4gp as sgp
        >>> import pandas
        >>> import json
        >>> dummy_data = [[30, 3, 1, 10], [35, 2, 2, 8], [40, 4, 2, 7], [50, 1, 1, 6], [52, 7, 1, 2]]
        >>> dummy_df = pandas.DataFrame(dummy_data, columns=['Age', 'Salary', 'Cars', 'Expenses'])
        >>>
        >>> mine_obj = sgp.GeneticGRAANK(dummy_df, 0.5, max_iter=3, n_pop=10)
        >>> result_json = mine_obj.discover()
        >>> result = json.loads(result_json)
        >>> # print(result['Patterns'])
        >>> print(result_json) # doctest: +SKIP
        {"Algorithm": "GA-GRAANK", "Best Patterns": [[["Age+", "Salary+", "Expenses-"], 0.6]], "Invalid Count": 12,
            "Iterations": 2}
        """
        super(GeneticGRAANK, self).__init__(*args, **kwargs)
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
        attr_keys = [GI(x[0], x[1].decode()).as_string for x in self.valid_bins[:, 0]]

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
