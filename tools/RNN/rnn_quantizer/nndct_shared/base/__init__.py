from .key_names import *
from .debug_level import *
GLOBAL_MAP = GlobalMap()


class SingletonMeta(type):
  _instances = {}
  
  def __call__(cls, *args, **kwargs):
    if cls not in cls._instances:
      cls._instances[cls] = super(SingletonMeta, cls).__call__(*args, **kwargs)
      cls._instance = cls._instances[cls]
    return cls._instances[cls]
      