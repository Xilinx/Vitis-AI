

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

class BaseCommander:

  @classmethod
  def register(cls, obj, attr):
    if not hasattr(obj, attr):
      setattr(obj, attr, {})
    commander = cls()
    getattr(obj, attr).update(commander.get_all_commands())

  @classmethod
  def get_all_commands(cls):
    all_commands = {}
    commander = cls()
    while True:
      try:
        all_commands.update({k:v for k,v in commander.create_commands().items() if \
            k!='self' and not k.startswith('_') and not k in all_commands})
        commander = super(commander.__class__, commander)
      except AttributeError:
        break
    return all_commands
