
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

from .miscellaneous import analyze_gps
from .miscellaneous import get_num_cores
from .miscellaneous import get_slurm_cores
from .miscellaneous import write_file

# Project Details
__version__ = "0.5.1"
__title__ = f"so4gp (v{__version__})"
__author__ = "Dickson Owuor"
__credits__ = "Montpellier University"
