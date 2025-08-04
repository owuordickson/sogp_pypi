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

        >>> import so4gp as sgp
        >>> import pandas
        >>> import json
        >>> dummy_data = [[30, 3, 1, 10], [35, 2, 2, 8], [40, 4, 2, 7], [50, 1, 1, 6], [52, 7, 1, 2]]
        >>> dummy_df = pandas.DataFrame(dummy_data, columns=['Age', 'Salary', 'Cars', 'Expenses'])
        >>>
        >>> mine_obj = sgp.ParticleGRAANK(dummy_df, 0.5, max_iter=3, n_particle=10)
        >>> result_json = mine_obj.discover()
        >>> result = json.loads(result_json)
        >>> # print(result['Patterns'])
        >>> print(result_json) # doctest: +SKIP
        {"Algorithm": "PSO-GRAANK", "Best Patterns": [], "Invalid Count": 12, "Iterations": 2}
        """
        super(ParticleGRAANK, self).__init__(*args, **kwargs)
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
        attr_keys = [GI(x[0], x[1].decode()).as_string for x in self.valid_bins[:, 0]]

        if self.valid_bins is None:
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
