

#
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
#

from collections import deque


def graph_search_handler(start_node,
                         generator,
                         frontier,
                         handler=None,
                         gen_params={}):
  frontier.append(start_node)
  explored = set()
  while frontier:
    node = frontier.pop()
    explored.add(node)
    if handler:
      handler(node)
    frontier.extend([
        n for n in generator(node, **gen_params)
        if n not in explored and n not in frontier
    ])
  return None


def breadth_first_search_handler(start_node,
                                 generator,
                                 handler=None,
                                 gen_params={}):
  return graph_search_handler(
      start_node,
      generator,
      frontier=FIFOQueue(),
      handler=handler,
      gen_params=gen_params)


class Queue:
  """Queue is an abstract class/interface. There are three types:
        Stack(): A Last In First Out Queue.
        FIFOQueue(): A First In First Out Queue.
        PriorityQueue(order, f): Queue in sorted order (default min-first).
    Each type supports the following methods and functions:
        q.append(item)  -- add an item to the queue
        q.extend(items) -- equivalent to: for item in items: q.append(item)
        q.pop()         -- return the top item from the queue
        len(q)          -- number of items in q (also q.__len())
        item in q       -- does q contain item?
    Note that isinstance(Stack(), Queue) is false, because we implement stacks
    as lists.  If Python ever gets interfaces, Queue will be an interface."""

  def __init__(self):
    raise NotImplementedError

  def extend(self, items):
    for item in items:
      self.append(item)


class FIFOQueue(Queue):
  """A First-In-First-Out Queue implemented with collections.deque
    
    MODIFIED FROM AIMA VERSION
        - Use deque
        - Use an additional dict to track membership
    """

  def __init__(self):
    self.A = deque()
    self.__keys = set()

  def append(self, item):
    self.A.append(item)
    self.__keys.add(item)

  def __len__(self):
    return len(self.A)

  def pop(self):
    key = self.A.popleft()
    self.__keys.discard(key)
    return key

  def __contains__(self, item):
    return item in self.__keys
