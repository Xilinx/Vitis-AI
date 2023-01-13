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

class Registry(object):
  """Provides a registry for saving objects."""

  def __init__(self, name):
    self._name = name
    self._registry = {}

  def register(self, obj, name):
    """Registers a Python object "obj" for the given "name".

    Args:
      obj: The object to add to the registry.
      name: The registered name for the obj.
    Raises:
      KeyError: If same name has been registered already.
    """
    if name in self._registry:
      raise KeyError("Name '{}' has been registered in '{}'".format(
                     (name, self._name)))

    self._registry[name] = obj

  def keys(self):
    """Returns a list of names of registered objects."""
    return self._registry.keys()

  def __contains__(self, name):
    return name in self._registry.keys()

  def lookup(self, name):
    """Looks up `name`.

    Args:
      name: a string specifying the registry key for the obj.
    Returns:
      Registered object if found
    Raises:
      LookupError: if `name` has not been registered.
    """
    if name in self._registry:
      return self._registry[name]
    else:
      raise LookupError('Registry has no entry for: {}'.format(name))

  def get(self, name, default_value):
    """Returns a registered object of given `name`. If there is no entry for
    given name, returns `default_value`.
    """
    return default_value if name not in self._registry else self._registry[name]
