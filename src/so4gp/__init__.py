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
@modified: 27 October 2022

SO4GP
------

    **SO4GP** stands for: "Some Optimizations for Gradual Patterns". SO4GP applies optimizations such as swarm
    intelligence, HDF5 chunks, SVD and many others to improve the efficiency of extracting gradual patterns
    (GPs). A GP is a set of gradual items (GI), and its quality is measured by its computed support value. A GI is a pair
    (i,v) where i is a column and v is a variation symbol: increasing/decreasing. Each column of a data set yields 2
    GIs; for example, column age yields GI age+ or age-. For example, given a data set with 3 columns (age, salary, cars)
    and 10 objects. A GP may take the form: {age+, salary-} with a support of 0.8. This implies that 8 out of 10 objects
    have the values of column age 'increasing' and column 'salary' decreasing.

    The classical approach for mining GPs is computationally expensive. This package provides Python algorithm
    implementations of several optimization techniques that are applied to the classical approach to improve
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


from .data_gp import DataGP
from .gradual_patterns import GI
from .gradual_patterns import GP
from .gradual_patterns import TGP
from .gradual_patterns import TimeDelay
from .gradual_patterns import PairwiseMatrix

from .utils import analyze_gps
from .utils import gradual_decompose
from .utils import get_num_cores
from .utils import get_slurm_cores
from .utils import write_file

# Project Details
__version__ = "0.7.0"
__title__ = f"so4gp (v{__version__})"
__author__ = "Dickson Owuor"
__credits__ = "Montpellier University"
