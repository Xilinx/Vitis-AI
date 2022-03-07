# Copyright 2019 Xilinx Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

""" collapse nodes within the namespace into a new specified node """

def _get_node_names(nodes):
  return [node.name for node in nodes]
# Given tensorflow GraphDef and a dict of namespace names -> plugin names,
# collapses those namespaces into single nodes representing plugins, excluding
# those nodes specified in exclude_nodes.
def _collapse_namespaces_impl(graph_def, namespace_map, exclude_node_names,
      unique_inputs, save_weights):
  nodes = graph_def.node
  # TODO: Maybe let this function arbitrarily collapse any group of nodes.
  # Will require more work on user end to collapse multiple namespaces if
  # implemented this way, but provides much greater flexibility. Maybe some
  # compromise is possible.

  def get_plugin_node(node_name):
    # Get the default plugin node provided by the user, or return None if this
    # does not belong in a plugin.
    if node_name in exclude_node_names:
      # Don't put this node into a plugin, treat as normal node instead.
      return None, None
    # Check if this node should be omitted from the main graph and return the plugin node if so.
    best_match_depth = -1
    best_match = None
    best_namespace = None
    for namespace in namespace_map:
      # Find the end point of the namespace
      current_depth = len(namespace.split('/'))
      # Get a section of the node path to the same depth
      node_namespace = "/".join(node_name.split('/')[:current_depth])
      # Try to match to the longest possible namespace path, then make sure it actually is a path.
      if namespace == node_namespace and current_depth > best_match_depth:
        best_match_depth = current_depth
        best_match = namespace_map[namespace]
        best_namespace = namespace
    return best_match, best_namespace

  def update_inputs(node):
    index = 0
    while index < len(node.input):
      input_name = node.input[index].replace('^', '')
      # We don't care if this is a control input for the purposes of plugins. (That's what the ^ indicates).
      input_plugin, _ = get_plugin_node(input_name)
      # If this input is in a plugin, replace with the plugin name instead.
      if input_plugin:
        # Remove and replace the node
        del node.input[index]
        if input_plugin.name not in node.input:
          # For plugin inputs, don't add duplicates.
          node.input.insert(index, input_plugin.name)
        else:
          index -= 1
      index += 1

  def update_plugin_inputs(plugin_node, node):
    def add_input(plugin_node, input_name):
      if not unique_inputs or input_name not in plugin_node.input:
        # If we're not checking for unique inputs, we can add the input all the time.
        # Otherwise, the input must not already be present.
        plugin_node.input.append(input_name)

    for input_name in node.input:
      # We don't care if this is a control input for the purposes of plugins. (That's what the ^ indicates).
      input_plugin, _ = get_plugin_node(input_name.replace('^', '').split(":")[0])
      # If the input is in a plugin, we need to add the plugin instead.
      if input_plugin:
        # If it's in the same plugin, it's not really an input; otherwise, we can add it.
        if input_plugin.name != plugin_node.name:
          add_input(plugin_node, input_plugin.name)
      else:
        # And if it's not in a plugin, just add it as a normal node.
        add_input(plugin_node, input_name)
    # copy original weights into plugin_node attr
    if save_weights and node.op == "Const":
      plugin_node.attr[node.name].CopyFrom(node.attr["value"])


  # Update the graph.
  index = 0
  while index < len(nodes):
    plugin_node, plugin_namespace = get_plugin_node(nodes[index].name)
    if plugin_node:
      # Add the inputs of this node to its plugin.
      update_plugin_inputs(namespace_map[plugin_namespace], nodes[index])
      # Finally, remove it from the main graph.
      del nodes[index]
      index -= 1
    else:
      # For non-plugin nodes, just update their inputs.
      update_inputs(nodes[index])
    index += 1

  # Then integrate the plugin nodes back into the graph.
  # NodeDef is an unhashable type.
  unique_nodes = []
  for node in namespace_map.values():
    if node not in unique_nodes:
      unique_nodes.append(node)

  nodes.extend(unique_nodes)
  return graph_def

# Wrapper to handle exclude_nodes
def collapse_namespaces(graph_def, namespace_map, exclude_nodes=[],
      unique_inputs=True, save_weights=False):
  '''
  Collapses nodes in namespaces to single nodes specified by the user, except where those nodes are marked for exclusion.

  Args:
      namespace_map (dict(str, tensorflow.NodeDef)): A dictionary specifying namespaces and their corresponding plugin nodes. These plugin nodes are typically used to specify attributes of the custom plugin, while inputs and outputs are automatically deduced. Multiple namespaces can be collapsed into a single plugin node, and nested namespaces are collapsed into plugin nodes outside their parent namespaces.
      exclude_nodes (list(tensorflow.NodeDef)): Iterable container (usually a list) of nodes which should NOT be collapsed. These nodes will be present in the final graph as either inputs or outputs of the plugin nodes.
      unique_inputs (bool): Whether inputs to the collapsed node should be unique. If this is false, plugin nodes may have duplicate inputs.

  Returns:
      None
  '''
  exclude_node_names = set(_get_node_names(exclude_nodes))
  return _collapse_namespaces_impl(graph_def, namespace_map,
          exclude_node_names, unique_inputs, save_weights)
