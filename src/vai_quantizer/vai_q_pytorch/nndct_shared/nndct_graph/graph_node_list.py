
import sys
from collections import Iterator, Iterable

FORWARD_DIR = 1
BACKWARD_DIR = 0
POSITION_UPPER_BOUND = sys.maxsize 
POSITION_LOWER_BOUND = - sys.maxsize - 1
APPEND_INTERVAL =  1099511627776 # 2^40
MID_POSITION = 0


class GraphNodeList(Iterable):
  def __init__(self, head, direction):
    self._head = head
    self._d = direction

  def __iter__(self):
    return GraphNodeListIterator(self._head, self._d)

class GraphNodeListIterator(object):
  def __init__(self, head, direction):
    self._head = head
    self._d = int(direction)
    self._next = None
    
  def __iter__(self):
    return self
  
  def __next__(self):
    if self._next is None:
      cur = self._head
      self._next = cur._neighbor_nodes[self._d]
      return cur
    
    if self._next is not self._head:
      cur = self._next
      self._next = self._next._neighbor_nodes[self._d]
      return cur
    else:
      raise StopIteration()
    