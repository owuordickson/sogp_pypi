

try:
    from . import DataGP, GI, ExtGP, TGP, TimeDelay
except ImportError:
    from src.so4gp import DataGP, GI, ExtGP, TGP, TimeDelay
