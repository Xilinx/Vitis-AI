# MIT License
#
# Copyright (c) 2023 Advanced Micro Devices, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import List

class NodeGroupUnion(object):
  """Utility class used to group nodes so that nodes in the same group
  will be pruned by the same `PruningSpec`.
  The process of grouping uses the Union-Find algorithm.
  See https://en.wikipedia.org/wiki/Disjoint-set_data_structure.
  """

  def __init__(self):
    # node_name => id
    self._node_id_map = {}
    self._parent = []
    self._group_size = []

  def add_node(self, name):
    if name in self._node_id_map:
      return

    node_id = len(self._parent)
    self._node_id_map[name] = node_id
    self._parent.append(node_id)
    self._group_size.append(1)

  def find(self, name):
    if name not in self._node_id_map:
      return -1

    node_id = self._node_id_map[name]
    # path compression
    root = node_id
    while root != self._parent[root]:
      root = self._parent[root]

    while node_id != self._parent[node_id]:
      parent = self._parent[node_id]
      self._parent[node_id] = root
      node_id = parent
    return root

  def union(self, x, y):
    """Union two sets by their sizes.
    Attach the tree with fewer elements to the root of tree having more elements.
    """
    x_root = self.find(x)
    y_root = self.find(y)
    if x_root == -1 or y_root == -1 or x_root == y_root:
      return

    # Swap x_root and y_root so that y_root represents the smaller group.
    if self._group_size[x_root] < self._group_size[y_root]:
      x_root, y_root = y_root, x_root

    # merge y_root to x_root
    self._parent[y_root] = x_root
    self._group_size[x_root] += self._group_size[y_root]

  def groups(self) -> List[List[str]]:
    groups = []
    root_to_group_index = {}
    for name in self._node_id_map:
      root = self.find(name)
      if root not in root_to_group_index:
        root_to_group_index[root] = len(groups)
        groups.append([])
      index = root_to_group_index[root]
      groups[index].append(name)
    return groups
