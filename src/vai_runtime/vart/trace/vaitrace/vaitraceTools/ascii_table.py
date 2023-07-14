#!/usr/bin/python3

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

def print_ascii_table(rows):
    headers = tuple(rows[0])

    lens = []

    def str_len(_x):
        return len(str(_x))

    for i in range(len(headers)):
        lens.append(
            str_len(max([x[i] for x in rows], key=lambda x: str_len(x))))

    formats = []
    hformats = []
    for i in range(len(headers)):
        formats.append("%%-%ds" % lens[i])
        hformats.append("%%-%ds" % lens[i])
    pattern = " | ".join(formats)
    hpattern = " | ".join(hformats)
    separator = "-+-".join(['-' * n for n in lens])

    table_width = len(hpattern % headers)
    print('=' * table_width)
    print(hpattern % headers)
    print(separator)

    def _u(t): return t
    for line in rows[1:]:
        print(pattern % tuple(_u(t) for t in line))

    print('=' * table_width)


class ascii_table:
    def __init__(_headers, _title=""):
        title = _title
        header = _headers
        data = []
        row_num = 1
        column_num = len(header)
        column_lens = []

        pass

    def add_data():
        pass

    def print():
        table_width = 0
        pass


if __name__ == "__main__":
    titles = ['SubGraph', 'Workload(GOPS)', 'AverageRunTime(ms)', 'Perf(GOPS)']
    data_1 = ["conv1",          123.333,    99.88, '694.2']
    data_2 = ['res2a_branch2a', 3129.099,   124.4, '-']
    data_3 = ["res2a_branch2b", 938.33,     32.33, '1101.0']
    data_4 = ["res2a_branch2c", 232.22,    32.3,  '676.8']
    data_5 = ["res3c_branch2c", 102.760,   33.5,  '556.6']

    """test rows"""
    print_ascii_table([titles, data_1, data_2, data_3, data_4, data_5])
