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
"""
This module contains some helper methods and classes used across other packages
in Neptune.
"""

import importlib
import json
import pkgutil
import tornado

def list_submodules(package, recursive=True):
    """
    Recursively (optional) find the submodules from a module or directory

    Args:
        package (str or module): Root module or directory to load submodules from
        recursive (bool, optional): Recursively find. Defaults to True.

    Returns:
        array: array containing module paths that can be imported
    """
    if isinstance(package, str):
        package = importlib.import_module(package)
    results = []
    for _loader, name, is_pkg in pkgutil.walk_packages(package.__path__):
        full_name = package.__name__ + '.' + name
        results.append(full_name)
        if recursive and is_pkg:
            results.extend(list_submodules(full_name))
    return results

def remove_prefix(text, prefix):
    """
    Removes the prefix string from the text

    Args:
        text (str): The base string containing the prefixed string
        prefix (str): Prefix to remove

    Returns:
        str: the base string with the prefix removed
    """
    if text.startswith(prefix):
        return text[len(prefix):]
    return text

class DummyClass(object):
    """
    This dummy class is used to hold attributes when destroying REST endpoints
    in Tornado (server.py, DestructServiceHandler). There's no Tornado API for
    this and the code from inside Tornado to do this expects an object
    containing the request.
    """
    pass

# https://stackoverflow.com/a/15721641
class MultiDimensionalArrayEncoder(json.JSONEncoder):
    """
    This JSON encoder transforms tuples (which are not JSON serializable) into
    arrays with a '__tuple__' key added. Then, when the JSON object is received,
    it can be parsed with the hinted_tuple_hook below to reconstruct the tuple.
    """
    def encode(self, obj):
        """
        The base encode method of the JSON encoder is called with one additional
        hook
        """
        def hint_tuples(item):
            """
            If the object to be serialized is a tuple, or contains tuples,
            replace tuples with a dictionary that can be reconstructed on
            the receiver

            Args:
                self (object): Object that will be JSON serialized

            Returns:
                object: Object without tuples
            """
            if isinstance(item, tuple):
                return {'__tuple__': True, 'items': item}
            if isinstance(item, list):
                return [hint_tuples(e) for e in item]
            if isinstance(item, dict):
                return {key: hint_tuples(value) for key, value in item.items()}
            return item

        return super(MultiDimensionalArrayEncoder, self).encode(hint_tuples(obj))

def hinted_tuple_hook(obj):
    """
    This hook can be passed to json.load* as a object_hook. Assuming the incoming
    JSON string is from the MultiDimensionalArrayEncoder, tuples are marked
    and then reconstructed into tuples from dictionaries.

    Args:
        obj (object): Object to be parsed

    Returns:
        object: Object (if it's a tuple dictionary, returns a tuple)
    """
    if '__tuple__' in obj:
        return tuple(obj['items'])
    return obj

def cancel_async_tasks():
    """
    Cancels pending tasks and closes the IO loop, which is needed in Python 3.
    This method is provided for Py2 compatibility and just closes the loop
    """
    tornado.ioloop.IOLoop.instance().stop()
