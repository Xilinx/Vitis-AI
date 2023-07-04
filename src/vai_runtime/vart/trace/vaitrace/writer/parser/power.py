
# Copyright 2022-2023 Advanced Micro Devices Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from writer.parser.tracepointUtil import *


def analyse_power(data):
    power_report = {}

    if len(data) == 0:
        return {}

    idle_power = min(data)
    peak_power = max(data)
    ave_power = sum(data) / len(data)

    power_report["idle_power"] = idle_power
    power_report["peak_power"] = peak_power
    power_report["ave_power"] = ave_power

    return power_report
