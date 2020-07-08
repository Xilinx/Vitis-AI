from nndct_shared.base import NNDCT_KEYS, GLOBAL_MAP

class ReachTheBaseExp(Exception):

  def __init__(self, cls):
    err = "Reach the Base of {}, no more items return".format(cls)
    Exception.__init__(self, err)

class RebuildMismatchError(Exception):

  def __init__(self, file_name=None):
    file_name = file_name or GLOBAL_MAP.get_ele(
        NNDCT_KEYS.MODIFIER).nndct_prefix + '.py'
    Exception.__init__(
        self,
        "The rebuilt graph mismatch with original graph, please manually modify '{}' and run again"
        .format(file_name))

class DeprecatedAPIError(Exception):

  def __init__(self, api_name: str, class_type=None):
    if class_type and isinstance(class_type, str):
      err = 'The api "{api_name}" in {class_type} is deprecated'.format(
          api_name=api_name, class_type=class_type)
    else:
      err = 'The api "{api_name}" is deprecated'.format(api_name=api_name)
    Exception.__init__(self, err)

class AddXopError(Exception):

  def __init__(self, op_name: str, op_type: str, msg: str):
    Exception.__init__(
        self, "Failed to add op(name:{}, type:{}) in xGraph!: '{}'".format(
            op_name, op_type, msg))

class ExportXmodelError(Exception):

  def __init__(self, graph_name):
    Exception.__init__(
        self, "Failed to convert nndct graph({}) to xmodel!".format(graph_name))

class DefineOptionError(Exception):

  def __init__(self, option_name, msg):
    Exception.__init__(
        self, "Option '{option_name}' initiate failed '{detail_msg}'!".format(
            option_name=option_name, detail_msg=msg))
