"""
 Copyright 2019 Xilinx Inc.

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

from functools import wraps
from contextlib import contextmanager
import time


def timefunc(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.clock()
        r = func(*args, **kwargs)
        end = time.clock()
        print("{}.{}: {} seconds".format(func.__module__, func.__name__, end - start))
        return r

    return wrapper


@contextmanager
def timeblock(label):
    start = time.clock()
    try:
        yield
    finally:
        end = time.clock()
        print("{} : {} seconds".format(label, end - start))

