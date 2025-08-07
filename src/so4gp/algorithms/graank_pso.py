# -*- coding: utf-8 -*-
# SPDX-License-Identifier: GNU GPL v3
# This file is licensed under the terms of the GNU GPL v3.0.
# See the LICENSE file at the root of this
# repository for complete details.


import json
import random
import numpy as np
from ..data_gp import DataGP
from .numeric_ss import NumericSS


class ParticleGRAANK(DataGP):

    def __init__(self, *args, max_iter: int = 1, n_particle: int = 5, vel: float = 0.9,
                 coeff_p: float = 0.01, coeff_g: float = 0.9, **kwargs):
        """Description

        Extract gradual patterns (GPs) from a numeric data source using the Particle Swarm Optimization Algorithm
        approach (proposed in a published research paper by Dickson Owuor). A GP is a set of gradual items (GI), and its
        quality is measured by its computed support value. For example, given a data set with 3 columns (age, salary,
        cars) and 10 objects. A GP may take the form: {age+, salary-} with a support of 0.8. This implies that 8 out of
        10 objects have the values of column age 'increasing' and column 'salary' decreasing.

            In this approach, it is assumed that every GP candidate may be represented as a particle that has a unique
            position and fitness. The fitness is derived from the computed support of that candidate, the higher the
            support value, the higher the fitness. The aim of the algorithm is to search through a population of particles
            (or candidates) and find those with the highest fitness as efficiently as possible.

        :param args: [required] data source path of Pandas DataFrame, [optional] minimum-support, [optional] eq
        :param max_iter: [optional] maximum_iteration, default is 1
        :param n_particle: [optional] initial particle population, default is 5
        :param vel: [optional] velocity, default is 0.9
        :param coeff_p: [optional] personal coefficient, default is 0.01
        :param coeff_g: [optional] global coefficient, default is 0.9

        >>> from so4gp.algorithms import ParticleGRAANK
        >>> import pandas
        >>> import json
        >>> dummy_data = [[30, 3, 1, 10], [35, 2, 2, 8], [40, 4, 2, 7], [50, 1, 1, 6], [52, 7, 1, 2]]
        >>> dummy_df = pandas.DataFrame(dummy_data, columns=['Age', 'Salary', 'Cars', 'Expenses'])
        >>>
        >>> mine_obj = ParticleGRAANK(dummy_df, 0.5, max_iter=3, n_particle=10)
        >>> result_json = mine_obj.discover()
        >>> result = json.loads(result_json)
        >>> # print(result['Patterns'])
        >>> print(result_json) # doctest: +SKIP
        {"Algorithm": "PSO-GRAANK", "Best Patterns": [], "Invalid Count": 12, "Iterations": 2}
        """
        super(ParticleGRAANK, self).__init__(*args, **kwargs)
        self._max_iteration: int = max_iter
        self._n_particles: int = n_particle
        self._velocity: float = vel
        self._coeff_p: float = coeff_p
        self._coeff_g: float = coeff_g

    def discover(self):
        """Description

        Searches through particle positions to find GP candidates. The candidates are validated if their computed
        support is greater than or equal to the minimum support threshold specified by the user.

        :return: JSON object
        """

        # Prepare data set
        self.fit_bitmap()
        self.clear_gradual_patterns()
        if self.valid_bins is None:
            return []

        # Initialize search space
        s_space = NumericSS.initialize_search_space(self.valid_bins, self._n_particles, self._max_iteration)
        if s_space is None:
            return []

        pbest_pop = s_space.pop.copy()
        gbest_particle = pbest_pop[0]
        velocity_vector = np.ones(self._n_particles)
        repeated = 0
        while s_space.counter < self._max_iteration:
            # while eval_count < max_evaluations:
            # while repeated < 1:
            for i in range(self._n_particles):
                if s_space.pop[i].position < s_space.var_min or s_space.pop[i].position > s_space.var_max:
                    s_space.pop[i].cost = 1
                else:
                    s_space.pop[i].cost = NumericSS.cost_function(s_space.pop[i].position, self.valid_bins)
                    if s_space.pop[i].cost == 1:
                        s_space.invalid_count += 1
                    s_space.eval_count += 1

                if pbest_pop[i].cost > s_space.pop[i].cost:
                    pbest_pop[i].cost = s_space.pop[i].cost
                    pbest_pop[i].position = s_space.pop[i].position

                if gbest_particle.cost > s_space.pop[i].cost:
                    gbest_particle.cost = s_space.pop[i].cost
                    gbest_particle.position = s_space.pop[i].position
            # if abs(gbest_fitness_value - self.target) < self.target_error:
            #    break
            if s_space.best_sol.cost > gbest_particle.cost:
                s_space.best_sol = NumericSS.Candidate(position=gbest_particle.position, cost=gbest_particle.cost)

            for i in range(self._n_particles):
                new_velocity = (self._velocity * velocity_vector[i]) + \
                               (self._coeff_p * random.random()) * (pbest_pop[i].position - s_space.pop[i].position) + \
                               (self._coeff_g * random.random()) * (gbest_particle.position - s_space.pop[i].position)
                s_space.pop[i].position = s_space.pop[i].position + new_velocity

            _, repeated = NumericSS.evaluate_gradual_pattern(self._max_iteration, repeated, s_space, self)
            
        # Output
        out = json.dumps({"Algorithm": "PSO-GRAANK", "Best Patterns": s_space.str_best_gps, 
                          "Invalid Count": s_space.invalid_count, "Iterations": s_space.iter_count})
        """:type out: object"""
        for gp in s_space.best_patterns:
            self.add_gradual_pattern(gp)
        return out
