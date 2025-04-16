# -*- coding: utf-8 -*-
# SPDX-License-Identifier: GNU GPL v3
# This file is dual licensed under the terms of the GNU GPL v3.0.
# See the LICENSE file in the root of this
# repository for complete details.


from .data_gp import DataGP
from .gradual_patterns import ExtGP
from .gradual_patterns import GI
from .gradual_patterns import GP
from .gradual_patterns import TGP
from .gradual_patterns import TimeDelay

from .so4gp import ClusterGP
from .so4gp import GRAANK
from .so4gp import AntGRAANK
from .so4gp import GeneticGRAANK
from .so4gp import HillClimbingGRAANK
from .so4gp import NumericSS
from .so4gp import ParticleGRAANK
from .so4gp import RandomGRAANK
from .so4gp import TGrad
from .so4gp import TGradAMI
from .so4gp import GradPFS

from .miscellaneous import analyze_gps
from .miscellaneous import gradual_decompose
from .miscellaneous import get_num_cores
from .miscellaneous import get_slurm_cores
from .miscellaneous import write_file

# Project Details
__version__ = "0.6.9"
__title__ = f"so4gp (v{__version__})"
__author__ = "Dickson Owuor"
__credits__ = "Montpellier University"
