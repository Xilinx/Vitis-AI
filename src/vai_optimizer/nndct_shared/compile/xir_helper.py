
import re
from collections import defaultdict
from .xgraph import _XMODEL_NAME_PATTERN

class XIRHelper(object):
  
  @classmethod
  def find_xops_from_nndct_node(cls, nndct_node, xmodel):
    xop_lst = []
    formal_name = re.sub(_XMODEL_NAME_PATTERN, "_", nndct_node.name)
    for xop in cls.get_xmodel_ops(xmodel):
      if cls.get_xop_type(xop) in ["download", "upload", "fix2float", "float2fix", "transpose", "fix", "data-fix"]:
        continue
      if formal_name in cls.get_xop_name(xop):
        xop_lst.append(xop)

    return xop_lst

  @staticmethod
  def get_xop_device_type(xop):
    if xop.has_attr("device"):
      return xop.get_attr("device")
    else:
      return None
    


  @staticmethod
  def get_xop_name(xop):
    return xop.get_name()

  @staticmethod
  def get_xop_template_name(op_template):
    return op_template.get_name()

  @staticmethod
  def get_xop_template_types(op_template):
    return op_template.get_types()

  @staticmethod
  def get_xmodel_ops(xmodel):
    return xmodel.get_ops()
  
  @staticmethod
  def get_xop_type(xop):
    return xop.get_type()

  @staticmethod
  def get_input_xops(xop):
    return xop.get_input_ops()["input"]
  
  @staticmethod
  def get_op_partition_msg(xop):
    msg = ""
    if xop and xop.has_attr("partition_msg"):
      msg = xop.get_attr("partition_msg")
    elif xop and xop.has_attr("error_msg"):
      msg = xop.get_attr("error_msg")
    return msg

  @classmethod
  def is_dpu_pattern(cls, xmodel):
    for xop in cls.get_xmodel_ops(xmodel):
      if cls.get_xop_device_type(xop) == "CPU":
        if cls.get_xop_type(xop) == "reshape-fix":
          input_op = cls.get_input_xops(xop)[0]
          if cls.get_xop_type(input_op) not in ["data", "data-fix"]:
            return False
        elif cls.get_xop_type(xop) not in ["fix2float", "download"]:
          return False
      elif cls.get_xop_device_type(xop) is None:
        return False
    return True

  @classmethod
  def get_pattern_partition_msg(cls, xmodel):
    msg = ""
    for xop in cls.get_xmodel_ops(xmodel):
      msg += cls.get_op_partition_msg(xop)
    return msg

  
  @classmethod
  def is_valid_compiled_pattern(cls, xmodel):
    for xop in cls.get_xmodel_ops(xmodel):
      if xop is None or xop.has_attr("error_msg"):
        return False
    if any([cls.get_xop_device_type(xop) is None for xop in cls.get_xmodel_ops(xmodel)]):
      return False
    return True


  



  

