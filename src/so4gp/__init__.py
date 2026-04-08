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
__version__ = "0.7.4"
__title__ = f"so4gp (v{__version__})"
__author__ = "Dickson Owuor"
__credits__ = "Montpellier University"
