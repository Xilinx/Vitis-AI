# Copyright 2019 Xilinx, Inc.
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

import json


class Profile:
    def __init__(self):
        self.m_profile = dict()

    def loadProfile(self, filePath):
        with open(filePath, 'r') as fh:
            self.m_profile = json.loads(fh.read())

    def writeProfile(self, filePath):
        with open(filePath, 'w') as fh:
            fh.write(json.dumps(self.m_profile, indent=2))


def main():

    profile = Profile()

    profile.m_profile['b_csim'] = True
    profile.m_profile['b_synth'] = True
    profile.m_profile['b_cosim'] = True

    profile.m_profile['dataTypes'] = [
        ('float', 64), ('float', 32), ('int', 16), ('int', 8)]

    profile.m_profile['op'] = 'amax'

    profile.m_profile['parEntries'] = 4
    profile.m_profile['vectorSizes'] = [128, 8192]
    profile.m_profile['valueRange'] = [-1024, 1024]
    profile.m_profile['numSimulation'] = 2

    profile.writeProfile(r'profile.json')


if __name__ == '__main__':
    main()
