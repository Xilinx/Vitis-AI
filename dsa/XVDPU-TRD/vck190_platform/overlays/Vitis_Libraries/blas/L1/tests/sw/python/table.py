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


import time


def list2File(
        listOfDict: list,
        filePath,
        addInfo=None,
        flag='a+',
        TabSize=2,
        Time=True):
    if len(listOfDict) == 0:
        return
    features = listOfDict[0]
    keys = features.keys()
    vlens = [[len(key)] + [len(str(f[key])) for f in listOfDict]
             for key in keys]
    maxLens = [max(len) + TabSize for len in vlens]

    delimiter = '+' + '+'.join(['-' * l for l in maxLens]) + '+\n'
    rowLen = len(delimiter)

    with open(filePath, flag) as f:
        f.write('=' * rowLen + '\n')
        if Time:
            f.write(time.ctime() + '\n')
        if addInfo is not None:
            f.write(addInfo)

        f.write(delimiter)
        ########### START OF KEYS ################
        strList = ['|']
        for s, l in zip(keys, maxLens):
            strList.append(('{:<%d}|' % l).format(s))
        strList.append('\n')
        f.write(''.join(strList))
        f.write(delimiter)
        ########### END OF KEYS ################

        ########### START OF VALUES ################
        for features in listOfDict:
            strList = ['|']
            for key, l in zip(keys, maxLens):
                strList.append(('{:<%d}|' % l).format(features[key]))
            strList.append('\n')
            f.write(''.join(strList))
            f.write(delimiter)
        ########### END OF VALUES ################
        f.write('=' * rowLen + '\n')
