"""Registry mechanism for "registering" classes/functions for general use.

This is typically used with a decorator that calls Register for adding
a class or function to a registry.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

class Registry(object):
  """Provides a registry for saving objects."""

  def __init__(self, name):
    """Creates a new registry."""
    self._name = name
    self._registry = {}

  def register(self, obj, name=None):
    """Registers a Python object "obj" for the given "name".

    Args:
      obj: The object to add to the registry.
      name: An optional string specifying the registry key for the obj.
            If None, obj.__name__ will be used.
    Raises:
      KeyError: If same name is registered twice.
    """
    if not name:
      name = obj.__name__
    if name in self._registry:
      raise KeyError("Name '%s' has been registered in '%s'!" %
                     (name, self._name))

    # logging.vlog(1, "Registering %s (%s) in %s.", name, obj, self._name)
    self._registry[name] = obj

  def list(self):
    """Lists registered items.

    Returns:
      A list of names of registered objects.
    """
    return self._registry.keys()

  def lookup(self, name):
    """Looks up "name".

    Args:
      name: a string specifying the registry key for the obj.
    Returns:
      Registered object if found
    Raises:
      LookupError: if "name" has not been registered.
    """
    if name in self._registry:
      return self._registry[name]
    else:
      raise LookupError("%s registry has no entry for: %s" % (self._name, name))
