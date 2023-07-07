
import copy
import networkx as nx
from networkx.algorithms import is_isomorphic
from nndct_shared.base import NNDCT_OP
from nndct_shared.inspector.utils import build_xir_nndct_op_map, log_debug_info
from nndct_shared.compile.xir_helper import XIRHelper
from .graph import Graph, Node

_SIMULATION_PATTERNS = [
   {"name": "conv2d_fix_with_hardwish", 
    "nodes":[("weights", Node({"fix2float"})), 
            ("bias", Node({"fix2float"})),  
            ("input", Node({"fix2float"})), 
            ("conv2d", Node({"conv2d", "matmul", "scale"})), 
            ("conv2d_out", Node({"float2fix"})),
            ("hsigmoid_in", Node({"fix2float"})),
            ("hsigmoid", Node({"hard-sigmoid"})),
            ("mul", Node({"mul"})),
            ("hsigmoid_out", Node({"float2fix"})),
            ("hswish_i0", Node({"fix2float"})),
            ("hswish_i1", Node({"fix2float"})),
            ("hswish", Node({"mul"})),
            ("output", Node({"float2fix"})),
            ],
  "edges": [("weights", "conv2d"), 
            ("bias", "conv2d"), 
            ("input", "conv2d"), 
            ("conv2d", "conv2d_out"), 
            ("conv2d_out", "hsigmoid_in"),
            ("hsigmoid_in", "hsigmoid"),
            ("hsigmoid", "mul"),
            ("mul", "hsigmoid_out"),
            ("conv2d_out", "hswish_i0"),
            ("hsigmoid_out", "hswish_i1"),
            ("hswish_i0", "hswish"),
            ("hswish_i1", "hswish"),
            ("hswish", "output"),
            ]
  },
  {"name": "conv2d_fix_with_hardsigmoid", 
   "nodes":[("weights", Node({"fix2float"})), 
            ("bias", Node({"fix2float"})),  
            ("input", Node({"fix2float"})), 
            ("conv2d", Node({"conv2d", "matmul", "scale"})), 
            ("conv2d_out", Node({"float2fix"})),
            ("hsigmoid_in", Node({"fix2float"})),
            ("hsigmoid", Node({"hard-sigmoid"})),
            ("mul", Node({"mul"})),
            ("output", Node({"float2fix"}))
            ],
  "edges": [("weights", "conv2d"), 
            ("bias", "conv2d"), 
            ("input", "conv2d"), 
            ("conv2d", "conv2d_out"), 
            ("conv2d_out", "hsigmoid_in"), 
            ("hsigmoid_in", "hsigmoid"), 
            ("hsigmoid", "mul"), 
            ("mul", "output")]
  },
  {"name": "conv2d_fix_with_relu", 
   "nodes":[("weights", Node({"fix2float"})), 
            ("bias", Node({"fix2float"})),  
            ("input", Node({"fix2float"})), 
            # ("conv2d", Node({"conv2d", "matmul", "depthwise-conv2d", "transposed-conv2d", "transposed-depthwise-conv2d", "scale"})), 
            ("conv2d", Node({"matmul", "scale"})), 
            ("relu", Node({"relu", "prelu", "leaky-relu", "relu6"})),
            ("output", Node({"float2fix"}))
            ],
  "edges": [("weights", "conv2d"), ("bias", "conv2d"), ("input", "conv2d"), ("conv2d", "relu"), ("relu", "output")]
  },
  {"name": "conv2d_fix_without_relu", 
   "nodes": [("weights", Node({"fix2float"})), 
            ("bias", Node({"fix2float"})),  
            ("input", Node({"fix2float"})), 
            # ("conv2d", Node({"conv2d", "matmul", "depthwise-conv2d", "transposed-conv2d", "scale"})),
            ("conv2d", Node({"matmul", "scale"})), 
            ("output", Node({"float2fix"}))
            ],
  "edges": [("weights", "conv2d"), ("bias", "conv2d"), ("input", "conv2d"), ("conv2d", "output")]
  },

  { 
    "name": "reduction_mean_with_mul_relu",
    "nodes": [
        ("input", Node({"fix2float"})),
        ("reduction_mean", Node({"reduction_mean"})),
        ("const", Node({"const"})),
        ("mul", Node({"mul"})),
        ("relu", Node({"relu","prelu", "leaky-relu", "relu6"})),
        ("output", Node({"float2fix"})),
      ],
      "edges": [("input", "reduction_mean"), ("reduction_mean", "mul"), ("const", "mul"), ("mul", "relu"), ("relu", "output")]

  },
  {
    "name": "reduction_mean_with_mul",
    "nodes": [
      ("input", Node({"fix2float"})),
      ("reduction_mean", Node({"reduction_mean"})),
      ("const", Node({"const"})),
      ("mul", Node({"mul"})),
      ("output", Node({"float2fix"})),
    ],
    "edges": [("input", "reduction_mean"), ("reduction_mean", "mul"), ("const", "mul"), ("mul", "output")]
  },
  # {
  #   "name": "reduction_mean_with_relu",
  #   "nodes": [
  #     ("input", Node({"fix2float"})),
  #     ("reduction_mean", Node({"reduction_mean"})),
  #     ("relu", Node({"relu","prelu", "leaky-relu", "relu6"})),
  #     ("output", Node({"float2fix"})),
  #   ],
  #   "edges": [("input", "reduction_mean"), ("reduction_mean", "relu"), ("relu", "output")]
  # },

  # {"name": "pool_with_mul", 
  #  "nodes": [("input", Node({"fix2float"})), 
  #           ("pool", Node({"avgpool2d"})),  
  #           ("const", Node({"const"})),
  #           ("mul", Node({"mul"})),
  #           ("output", Node({"float2fix"}))
  #           ],
  # "edges": [("input", "pool"), ("pool", "mul"), ("const", "mul"), ("mul", "output")]
  # },

  # {"name": "pool_fix", 
  #  "nodes": [("input", Node({"fix2float"})), 
  #           ("pool", Node({"avgpool2d", "maxpool2d"})),  
  #           ("output", Node({"float2fix"}))
  #           ],
  # "edges": [("input", "pool"), ("pool", "output")]
  # },
 
  # { "name": "eltwise_fix_with_relu", 
  #   "nodes": [("add", Node({"add", "mul"})), 
  #            ("relu", Node({"relu"})),  
  #            ("output", Node({"float2fix"}))
  #           ],
  #   "edges": [("add", "relu"), ("relu", "output")]
  # },
  # { "name": "eltwise_fix", 
  #   "nodes": [("add", Node({"add", "mul"})), 
  #             ("output", Node({"float2fix"}))
  #           ],
  #   "edges": [("add", "output")]
  # },


  # { "name": "concat_fix", 
  #   "nodes": [("concat", Node({"concat"})), 
  #             ("output", Node({"float2fix"}))
  #           ],
  #   "edges": [("concat", "output")]
  # },

  # {
  #   "name": "resize_fix",
  #   "nodes": [("input", Node({"fix2float"})),
  #             ("resize", Node({"resize"})),
  #             ("output", Node({"float2fix"}))
  #   ],

  #   "edges": [("input", "resize"), ("resize", "output")]

  # },

  # {
  #   "name": "pad_fix",
  #   "nodes": [("input", Node({"fix2float"})),
  #             ("pad", Node({"pad"})),

  #             ("output", Node({"float2fix"}))
  #           ],
  #   "edges": [("input", "pad"), ("pad", "output")]

  # },

  # {
  #   "name": "reduction_max_fix",
  #   "nodes": [("input", Node({"fix2float"})),
  #             ("reduction_max", Node({"reduction_max"})),

  #             ("output", Node({"float2fix"}))
  #           ],
  #   "edges": [("input", "reduction_max"), ("reduction_max", "output")],

  # },

  # {
  #   "name": "reshape_fix",
  #   "nodes": [("input", Node({"fix2float"})),
  #             ("reshape", Node({"reshape"})),

  #             ("output", Node({"float2fix"}))
  #           ],
  #   "edges": [("input", "reshape"), ("reshape", "output")],
  # },

  # {
  #   "name": "hsigmoid_fix",
  #   "nodes": [("input", Node({"fix2float"})),
  #             ("hsigmoid", Node({"hard-sigmoid"})),
  #             ("output", Node({"float2fix"}))
  #           ],
  #   "edges": [("input", "hsigmoid"), ("hsigmoid", "output")],
  # },

  # {
  #   "name": "reduction_mean",
  #   "nodes": [("input", Node({"fix"})),
  #             ("reduction_mean", Node({"reduction_mean"})),
  #             ("output", Node({"fix"}))
  #           ],
  #   "edges": [("input", "reduction_mean"), ("reduction_mean", "output")],
  # },

  # {
  #   "name": "reduce_max",
  #   "nodes": [("input", Node({"fix"})),
  #             ("poollikeop", Node({"reduction_max"})),
  #             ("output", Node({"fix"}))
  #           ],
  #   "edges": [("input", "poollikeop"), ("poollikeop", "output")],
  # },

]


def _gen_pattern_from_sim_pattern():
  pattern_graphs = []
  patterns = copy.deepcopy(_SIMULATION_PATTERNS)
  for pattern_info in patterns:
    pattern_graph = Graph(pattern_info["name"])
    for id, attr in pattern_info["nodes"]:
      pattern_graph.add_node(id, attr)
    
    for u, v in pattern_info["edges"]:
      pattern_graph.add_edge(u, v)
    pattern_graphs.append(pattern_graph)
  return pattern_graphs


def get_input_ops(op):
  return op.get_input_ops()

def get_output_ops(op):
  return op.get_fanout_ops()

def get_op_name(op):
  return op.get_name()

def get_op_type(op):
  return op.get_types()

def get_templates_from_dpu_compiler():
  """
    get pattern info from compiler
  """
  import xir
  import xcompiler
  templates = xcompiler.get_templates()
  topsorted_templates = []
  for template in templates:
    topsorted_templates.append((template.get_name(), [op for op in template.toposort()]))
  return topsorted_templates


def is_valid_pattern(pattern):
  fix = {"fix"}
  float2fix = {"float2fix"}
  fix2float = {"fix2float"} 
  argmax = {"argmax"}
  data = {"data"}
  const = {"const"}
  op_types = set()
  for node in pattern.nodes:
    op_types.update(pattern.get_node_types(node))
  
  if fix.issubset(op_types):
    msg = "This is a transfer pass template."
    return False, msg
  elif not (fix2float.issubset(op_types) or float2fix.issubset(op_types)):
    msg = "There is no fix in template."
    return False, msg
  elif all([{op_type} in [fix2float, float2fix] for op_type in op_types]):
    msg = "Only fix in template"
    return False, msg
  elif argmax.issubset(op_types):
    msg = "argmax template is ignored."
    return False, msg
  elif len(list(pattern.nodes)) == 2 and (data.issubset(op_types) or const.issubset(op_types)):
    msg = "data-fix/const-fix are ignored."
    return False, msg

  if not nx.algorithms.is_directed_acyclic_graph(pattern.graph):
    msg = f"{pattern.name} has cycles, please contact developer to fix it."
    return False, msg
  
  return True, ""
  

def is_valid_template(ops):
  fix = {"fix"}
  float2fix = {"float2fix"}
  fix2float = {"fix2float"} 
  argmax = {"argmax"}
  data = {"data"}
  const = {"const"}
  false_msg = "This pattern is not for quantization."
  float_template = False
  op_types = set()
  for op in ops:
    op_types.update(XIRHelper.get_xop_template_types(op))
  
  if fix.issubset(op_types):
    msg = "This is a transfer pass template."
    return False, msg
  elif not (fix2float.issubset(op_types) or float2fix.issubset(op_types)):
    msg = "There is no fix in template."
    return False, msg
  elif all([{op_type} in [fix2float, float2fix] for op_type in op_types]):
    msg = "Only fix in template"
    return False, msg
  elif argmax.issubset(op_types):
    msg = "argmax template is ignored."
    return False, msg
  elif len(ops) == 2 and (data.issubset(op_types) or const.issubset(op_types)):
    msg = "data-fix/const-fix are ignored."
    return False, msg
  return True, ""
  
    
def build_patterns_from_dpu_templates():
  templates = get_templates_from_dpu_compiler()
  log_debug_info("\nAll patterns from xcompiler:") 
  for id, (name, ops) in enumerate(templates):
    log_debug_info(f"pattern id:{id}")
    for op in ops:
      log_debug_info(f"op name:{XIRHelper.get_xop_template_name(op)} type:{XIRHelper.get_xop_template_types(op)}")
  
  patterns = []
  pattern_graphs = []
  for id, (name, ops) in enumerate(templates):
    pattern_graph = create_pattern_graph(f"{name}_{id}", ops)
    ret, msg = is_valid_pattern(pattern_graph)
    if ret:
      pattern_graphs.append(pattern_graph)
    else:
      log_debug_info(f"{pattern_graph.name} is filtered.({msg}).")

  # pattern_graphs = pattern_graphs + _gen_pattern_from_sim_pattern()
  log_debug_info("\nPattern Transformation:") 
  for pattern_graph in pattern_graphs:
    log_debug_info(f"{pattern_graph.name} pattern") 
    log_debug_info("================Before transformation====================")
    log_debug_info(str(pattern_graph))
    transform_pattern_graph(pattern_graph)
    if convert_xir_type_to_nndct_type(pattern_graph):
      patterns.append(pattern_graph)
    else:
      log_debug_info(f"{pattern_graph.name} is ignored for there is at least one unknown op in the pattern.")
    log_debug_info("================After transformation====================")
    log_debug_info(str(pattern_graph))
  patterns = reorder_patterns(patterns)
  return patterns


def reorder_patterns(patterns):
  new_patterns = []
  pattern_len_map = {}
  pattern_map = {pattern.name: pattern for pattern in patterns}

  fix_type = {NNDCT_OP.FIX}
  for pattern in patterns:
    pattern_len = 0
    for node in pattern.nodes:
      if pattern.get_node_types(node) != fix_type:
        pattern_len += 1

    pattern_len_map[pattern.name] = pattern_len
  
  sorted_patterns = sorted(pattern_len_map.items(), key=lambda x: x[1], reverse=True)
  log_debug_info(f"==============sorted patterns(total {len(sorted_patterns)} patterns)====================")
  for pattern_name, _, in sorted_patterns:
    log_debug_info(pattern_name)
    log_debug_info(str(pattern_map[pattern_name]))
  return [pattern_map[pattern_name] for pattern_name, _ in sorted_patterns]

def create_pattern_graph(name: str, ops: "List[xir.op_template]"):
  pattern_graph = Graph(name)
  for op in ops:
    pattern_graph.add_node(get_op_name(op), Node(op_types=get_op_type(op)))
  
  for op in ops:
    for inp in get_input_ops(op):
      pattern_graph.add_edge(get_op_name(inp), get_op_name(op))
    for outp in get_output_ops(op):
      pattern_graph.add_edge(get_op_name(op), get_op_name(outp))

  return pattern_graph

def _merge_float2fix_fix2float_pair(pattern_graph):
  float2fix = {"float2fix"}
  fix2float = {"fix2float"}
  node_pairs = []
  for node in pattern_graph.nodes:
    if pattern_graph.get_node_types(node) == float2fix and len(pattern_graph.children(node)) > 0 and all([pattern_graph.get_node_types(cn) == fix2float for cn in pattern_graph.children(node)]):
      node_pairs.append((node, [cn for cn in pattern_graph.children(node)]))

  for idx, (up_fix, down_fixes) in enumerate(node_pairs):
    pn_fix_0 = pattern_graph.parents(up_fix)[0]
    fix_node = Node({"fix"})
    fix_name = f"fix_{idx}"
    pattern_graph.add_node(fix_name, fix_node)
    pattern_graph.add_edge(pn_fix_0, fix_name)
    pattern_graph.remove_edge(pn_fix_0, up_fix)
    for fix in down_fixes:
      for cn in pattern_graph.children(fix):
        pattern_graph.add_edge(fix_name, cn)
        pattern_graph.remove_edge(fix, cn)
        
    pattern_graph.remove_one_node(up_fix)
    for fix in down_fixes:
      pattern_graph.remove_one_node(fix)

def _convert_fix_like_op_to_fix(pattern_graph):
  for node in pattern_graph.nodes:
    types = pattern_graph.get_node_types(node)
    types = list(types)
    for i, ty in enumerate(types):
      if ty in ["float2fix", "fix2float"]:
        types[i] = "fix"
    types = set(types)
    pattern_graph.set_node_types(node, types)


def _merge_mul_coeff(pattern_graph):
  ops_with_coeff = {"avgpool2d", "hard-sigmoid", "reduction_mean"}
  mul = {"mul"}
  const = {"const"}
  merged_node = []
  if not (ops_with_coeff & pattern_graph.op_types):
    return 

  for node in pattern_graph.nodes:
    if pattern_graph.get_node_types(node).issubset(ops_with_coeff) and len(pattern_graph.children(node)) == 1 and pattern_graph.get_node_types(pattern_graph.children(node)[0]) == mul:
      merged_node.append(pattern_graph.children(node)[0])

  for node in merged_node:
    in_nodes = pattern_graph.parents(node)
    out_nodes = pattern_graph.children(node)
    for pn in in_nodes:
      pattern_graph.remove_edge(pn, node)
      if pattern_graph.get_node_types(pn) == const:
        pattern_graph.remove_node(pn)
      else:
        if out_nodes:
          for cn in out_nodes:
            pattern_graph.add_edge(pn, cn)
    pattern_graph.remove_node(node)


def _remove_mul_for_hswish(pattern_graph):
  hsigmoid = {"hard-sigmoid"}
  hswish = {"hard-swish"}
  mul = {"mul"}
  fix = {"fix"}
  hswish_types = set()
  hswish_types.update(hsigmoid)
  hswish_types.update(mul)
  hswish_types.update(fix)
  merged_node = []
  graph_template =  {"name": "hardwish", 
    "nodes":[("weights", Node({"fix"})), 
            ("input", Node({"fix"})), 
            ("conv2d", Node({"conv2d", "matmul", "scale"})), 
            ("conv2d_out", Node({"fix"})),
            ("hsigmoid", Node({"hard-sigmoid"})),
            ("hsigmoid_out", Node({"fix"})),
            ("hswish", Node({"mul"})),
            ("output", Node({"float2fix"})),
            ],
  "edges": [("weights", "conv2d"), 
            ("input", "conv2d"), 
            ("conv2d", "conv2d_out"), 
            ("conv2d_out", "hsigmoid"),
            ("hsigmoid", "hsigmoid_out"),
            ("conv2d_out", "hswish"),
            ("hsigmoid_out", "hswish"),
            ("hswish", "output"),
            ]
  }
  
  hswish_pattern = Graph(graph_template["name"])
  for id, attr in graph_template["nodes"]:
    hswish_pattern.add_node(id, attr)
  
  for u, v in graph_template["edges"]:
    hswish_pattern.add_edge(u, v)
 
  if not hswish_types.issubset(pattern_graph.op_types) or (not is_isomorphic(pattern_graph.graph, hswish_pattern.graph)):
    return  


  for node in pattern_graph.nodes:
    if pattern_graph.get_node_types(node) == hsigmoid:
      pattern_graph.set_node_types(node, hswish)
    elif pattern_graph.get_node_types(node) == mul:
      merged_node.append(node)
      
  for node in merged_node:
    in_nodes = pattern_graph.parents(node)
    out_nodes = pattern_graph.children(node)
    for pn in in_nodes:
      pattern_graph.remove_edge(pn, node)
    for cn in out_nodes:
      pattern_graph.remove_edge(node, cn)
      pattern_graph.remove_node(cn)
    pattern_graph.remove_node(node)

_, _XIR2NNCT = build_xir_nndct_op_map()
_XIR2NNCT["fix"] = {NNDCT_OP.FIX}


def convert_xir_type_to_nndct_type(pattern_graph):
  for node in pattern_graph.nodes:
    xir_types = pattern_graph.get_node_types(node)
    nndct_types = set()
    for ty in xir_types:
      if ty in _XIR2NNCT:
        nndct_types.update(_XIR2NNCT.get(ty, {ty}))
    if nndct_types:
      pattern_graph.set_node_types(node, nndct_types)
    else:
      return False
  return True


def transform_pattern_graph(pattern_graph):
  _merge_float2fix_fix2float_pair(pattern_graph)
  _convert_fix_like_op_to_fix(pattern_graph)
  _merge_mul_coeff(pattern_graph)
  _remove_mul_for_hswish(pattern_graph)


def drop_fix_in_pattern(pattern_graph):
  fix = {NNDCT_OP.FIX}
  removed_node = []
  pattern_without_fix = Graph.copy(pattern_graph)
  for node in pattern_graph.nodes:
    if pattern_without_fix.get_node_types(node) == fix:
      removed_node.append(node)

  for node in removed_node:
    pattern_without_fix.remove_one_node(node)
  
  return pattern_without_fix




  
