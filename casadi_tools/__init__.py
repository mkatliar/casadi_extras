from .collocation import CollocationScheme, cheb
from .dae_model import Dae

from .graph import *
from .bounds import *

import sys
if sys.version_info >= (3,0):
  from .structure3 import repeated, entry, struct_symSX, struct_symMX, struct_SX, struct_MX, struct_MX_mutable, nesteddict, index, indexf, struct, struct_load
else:
  from .structure import repeated, entry, struct_symSX, struct_symMX, struct_SX, struct_MX, struct_MX_mutable, nesteddict, index, indexf, struct, struct_load
from .in_out import nice_stdout, capture_stdout

def print_subclasses(myclass, depth=0):
  print(("  " * depth) + " - " + myclass.__name__)
  for s in myclass.__subclasses__():
    print_subclasses(s,depth=depth+1)

def loadAllCompiledPlugins():
  for k in CasadiMeta.plugins().split(";"):
    cls, name = k.split("::")
    print("Testing: ", cls, name)
    if cls in ("Importer","XmlFile","Linsol"):
      getattr(casadi,cls).load_plugin(name)
    else:
      getattr(casadi,'load_'+cls.lower())(name)
