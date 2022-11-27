# Copyright 2022 Xilinx Inc.
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

from nndct_shared.utils import NndctScreenLogger

try:
  from vai_utf.python.target_factory import VAI_UTF as utf
except ModuleNotFoundError:
  NndctScreenLogger().warning("Can't find vai_utf package in your environment for inspector.")
except Exception as e:
  NndctScreenLogger().warning(f"Import module 'vai_utf' error: '{str(e)}'")
from .inspector import InspectorImpl