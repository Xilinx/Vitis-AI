import h5py
import json

from nndct_shared.nndct_graph.base_tensor import Tensor

def save_graph(nndct_graph, hdf5_path='graph.hdf5'):
  GraphHDF5Saver(nndct_graph).save(hdf5_path)

class GraphHDF5Saver():

  def __init__(self, nndct_graph):
    self.graph = nndct_graph

  def get_node_config(self, node):
    node_info = dict()
    node_info['idx'] = node.idx
    node_info['name'] = node.name
    node_info['dtype'] = str(node.dtype)

    for idx, tensor in enumerate(node.in_tensors):
      node_info['in_tensors{}.name'.format(idx)] = tensor.name
      node_info['in_tensors{}.shape'.format(idx)] = tensor.shape
      node_info['in_tensors{}.dtype'.format(idx)] = tensor.dtype

    for idx, tensor in enumerate(node.out_tensors):
      node_info['out_tensors{}.name'.format(idx)] = tensor.name
      node_info['out_tensors{}.shape'.format(idx)] = tensor.shape
      node_info['out_tensors{}.dtype'.format(idx)] = tensor.dtype

    for attr_enum, attr in node.op.attrs.items():
      if isinstance(attr.value, Tensor):
        continue
      elif isinstance(attr.value, (tuple, list)):
        has_tensor = False
        for val in attr.value:
          if isinstance(val, Tensor):
            has_tensor = True
            break
        if not has_tensor:
          node_info['Attr.{}'.format(attr_enum.name)] = attr.value
      else:
        node_info['Attr.{}'.format(attr_enum.name)] = attr.value
    return node_info

  def get_model_config(self):
    model_config = {'name': self.graph.name}
    model_config['layers'] = list()

    for node in self.graph.nodes:
      node_info = dict()
      node_info['class_name'] = node.op_type
      node_info['name'] = node.name
      node_info['inbound_nodes'] = [[[i, 0, 0, {}] for i in node.in_nodes]]
      node_info['config'] = self.get_node_config(node)
      model_config['layers'].append(node_info)

    return model_config

  def save(self, hdf5_path):
    config = self.get_model_config()

    model_config = {'class_name': 'Functional', 'config': config}
    metadata = dict(model_config=model_config)
    f = h5py.File(hdf5_path, mode='w')
    try:
      for k, v in metadata.items():
        if isinstance(v, (dict, list, tuple)):
          f.attrs[k] = json.dumps(v).encode('utf8')
        else:
          f.attrs[k] = v
      f.flush()
    finally:
      f.close()

class GraphConvertToCFG():
  # Visualizing network structure with multiple inputs is not supported.
  def __init__(self, nndct_graph, cfg_savepath='test.cfg'):
    self.graph = nndct_graph
    self.cfg_path = cfg_savepath
    self.content = []

  def read_node(self, node):
    self.content.append('name={}'.format(node.name))
    self.content.append('scope_name={}'.format(node.scope_name))
    self.content.append('idx={}'.format(str(node.idx)))
    self.content.append('dtype={}'.format(node.dtype))
    self.content.append('in_nodes={}'.format(str(node.in_nodes)))
    self.content.append('out_nodes={}'.format(str(node.out_nodes)))

    for idx, tensor in enumerate(node.in_tensors):
      self.content.append('in_tensors{}.name={}'.format(str(idx), tensor.name))
      self.content.append('in_tensors{}.shape={}'.format(
          str(idx), str(tensor.shape)))
      self.content.append('in_tensors{}.dtype={}'.format(
          str(idx), str(tensor.dtype)))

    for idx, tensor in enumerate(node.out_tensors):
      self.content.append('out_tensors{}.name={}'.format(str(idx), tensor.name))
      self.content.append('out_tensors{}.shape={}'.format(
          str(idx), str(tensor.shape)))
      self.content.append('out_tensors{}.dtype={}'.format(
          str(idx), str(tensor.dtype)))

    for name, attr in node.op.attrs.items():
      self.content.append('op_{}={}'.format(name, attr.value))

  def convert(self):
    # Traverse every node in graph sequentially once. And all input nodes of the current node must have been traversed before.
    index = 0
    nodename_index = {}
    last_nodename = None
    is_first_layer = True

    for node in self.graph.nodes:
      nodename = node.name
      op_type = node.op.type
      if is_first_layer:
        self.content.append('height={}'.format(node.out_tensors[0].shape[1]))
        self.content.append('width={}'.format(node.out_tensors[0].shape[2]))
        self.content.append('channels={}'.format(node.out_tensors[0].shape[3]))
        self.read_node(node)
        is_first_layer = False
        last_nodename = nodename
        continue
      num_innodes = len(node.in_nodes)
      nodename_index[nodename] = index
      index += 1
      if num_innodes == 0:
        self.content.append('[{}]'.format(op_type))
        self.read_node(node)
      elif num_innodes == 1:
        in_nodename = node.in_nodes[0]
        if in_nodename == last_nodename:
          self.content.append('[{}]'.format(op_type))
          self.read_node(node)
        else:
          self.content.append('[route]')
          self.content.append('layers={}'.format(
              str(nodename_index[in_nodename])))
          nodename_index[nodename] = index
          index += 1
          self.content.append('[{}]'.format(op_type))
          self.read_node(node)
      else:
        self.content.append('[route]')
        str_layers = 'layers='
        for i in range(len(node.in_nodes)):
          if i == 0:
            str_layers += str(nodename_index[node.in_nodes[i]])
          else:
            str_layers += ',' + str(nodename_index[node.in_nodes[i]])
        self.content.append(str_layers)
        self.content.append('op_type={}'.format(op_type))
        self.read_node(node)
      last_nodename = nodename

    with open(self.cfg_path, 'w') as f:
      f.write('[net]' + '\n')
      f.writelines(line + '\n' for line in self.content)
