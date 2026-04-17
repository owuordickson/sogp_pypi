from .data_gp import DataGP
from .gradual_patterns import GI
from .gradual_patterns import GP
from .gradual_patterns import TGP
from .gradual_patterns import TimeDelay
from .gradual_patterns import PairwiseMatrix

from .utils import get_num_cores
from .utils import get_slurm_cores
from .utils import write_file
from .utils import gen_gradual_warping_path


from functools import wraps
@wraps(DataGP.analyze_gps)
def analyze_gps(*args, **kwargs):
    return DataGP.analyze_gps(*args, **kwargs)

# Project Details
__version__ = "0.7.9"
__title__ = f"so4gp (v{__version__})"
__author__ = "Dickson Owuor"
__credits__ = "Montpellier University"


__all__ = [
    "DataGP",
    "GI",
    "GP",
    "TGP",
    "TimeDelay",
    "PairwiseMatrix",
    "get_num_cores",
    "get_slurm_cores",
    "write_file",
    "gen_gradual_warping_path",
    "analyze_gps"
]
