from .collocation import CollocationScheme, cheb, Pdq, collocationIntegrator
from .system_trajectory import SystemTrajectory
from .dae_model import SemiExplicitDae, ImplicitDae
from .inequality import Inequality, BoundedVariable

import sys
if sys.version_info >= (3,0):
  from .structure3 import repeated, entry, struct_symSX, struct_symMX, struct_SX, struct_MX, struct_MX_mutable, nesteddict, index, indexf, struct, struct_load
else:
  raise RuntimeError('Python3 is required')
