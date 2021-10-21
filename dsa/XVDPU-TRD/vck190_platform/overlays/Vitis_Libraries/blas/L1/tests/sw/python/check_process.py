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
import functools
import argparse
import os
import sys
import re
import subprocess
import shlex
import pdb


def Format(x):
    f_dic = {0: 'th', 1: 'st', 2: 'nd', 3: 'rd'}
    k = x % 10
    if k >= 4:
        k = 0
    return "%d%s" % (x, f_dic[k])


class Jobs:
    def __init__(self, jobList):
        self.jobIds = list()
        self.statList = list()
        self.size = 0
        pattern = r"Job <(\d+)> is submitted"
        for i in range(len(jobList)):
            file = jobList[i]
            with open(file, 'r') as f:
                content = f.read()
                match = re.search(pattern, content)
                if match is not None:
                    id = match.group(1)
                    self.jobIds.append(id)
                    self.statList.append('statistics_%d.rpt' % i)
                    self.size += 1

    def alive(self, id):
        commandLine = "bjobs %s" % id
        args = shlex.split(commandLine)
        result = subprocess.check_output(
            args, stderr=subprocess.STDOUT).decode("utf-8")
        pattern = "not found"
        match = re.search(pattern, result)
        if match is None:
            return True
        else:
            return False

    def aliveAny(self):
        a = [self.alive(id) for id in self.jobIds]
        def func(x, y): return x or y
        return functools.reduce(func, a)

    def checks(self):
        fileFound = [os.path.exists(f) for f in self.statList]
        def func(x, y): return x and y
        return functools.reduce(func, fileFound)


def poll(jobs, t, id_max, progress=80):
    id = 0
    alive = True
    while id < id_max or alive:
        id += 1

        if alive:
            id_max += 1

        if jobs.checks():
            print("Polling finished.")
            break

        print("Sleeping for %ds." % (t))
        perT = t / progress
        sys.stdout.write('%s: [=' % Format(id))
        for i in range(progress):
            sys.stdout.write('\b=%d' % (i % 10))
            sys.stdout.flush()
            time.sleep(perT)
        sys.stdout.write('\b]\n')
        alive = jobs.aliveAny()
    return id == id_max


def merge(fileList, filename):
    with open(filename, 'w+') as f:
        for file in fileList:
            with open(file, 'r+') as fr:
                f.write(fr.read())


def main(args):

    jobs = Jobs(['%s_%d' % ('job', i) for i in range(args.number)])
    if jobs.size == 0:
        return
    if poll(jobs, args.time, args.timeout / args.time):
        print("Time out, please check the logfiles.")
    else:
        merge(jobs.statList, 'statistics.rpt')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Generate random vectors and run test.')
    parser.add_argument(
        '--number',
        type=int,
        required=True,
        help='number of files')
    parser.add_argument('--time', type=int, default=60,
                        help='number of seconds to poll')
    parser.add_argument('--timeout', type=int, default=120,
                        help='number of seconds to time out')
    args = parser.parse_args()

    main(args)
