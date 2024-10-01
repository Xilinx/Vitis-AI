#!/usr/bin/env python

# Copyright 2022-2023 Advanced Micro Devices Inc.
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

import sys
import os.path
import xml.etree.ElementTree as ET

from python import Root

from functools import reduce


class mcInst:
    def __init__(self, inst):
        self.inst = inst

    def code_gen(self):
        inst = self.inst
        words = []
        reserved_id = 1
        reserved_name = "redvered"
        for w in inst.word_list:
            # uint32_t bank_addr : 14, bank_id : 6, dpby : 4, dpdon : 4, opcode : 4;
            f_code = []
            for f in w.field_list[::-1]:
                field_name = f.name
                if field_name == "reserved":
                    field_name = "%s_%d" % (reserved_name, reserved_id)
                    reserved_id += 1
                field_len = f.len
                f_code.append("{} : {}".format(field_name, field_len))

            f_code_str = reduce(lambda _x, _y: "{}, {}".format(_x, _y), f_code)
            words.append("uint32_t {};\n".format(f_code_str))
        code = "struct {} {{{}}};\n\n".format(
            inst.name, reduce(lambda _x, _y: _x + _y, words))

        return code


class InstTable:
    def __init__(self, insts):
        self.inst_table = insts

    def code_gen(self):
        code = "std::vector<class inst_desc> inst_table = {{{}}};\n"
        tmp = []
        for i in self.inst_table:
            ii = {"inst_name": i.name.upper(), "inst_opcode": i.opcode_str,
                  "inst_len": i.word_num}
            tmp.append(
                "create_inst_desc ({inst_name}, {inst_opcode}, {inst_len})".format(**ii))
        code = code.format(reduce(lambda _x, _y: "{},{}".format(_x, _y), tmp))

        return code


def gen_mc_header(root_list, dir="./"):

    for dpuInstance in root_list:
        h_f = "%s.h" % dpuInstance.version
        h = open(os.path.join(dir, h_f), "w+t")
        i_code = ""
        for i in dpuInstance.inst_list:
            i_code = mcInst(i).code_gen()
            h.write(i_code)

        t_code = InstTable(dpuInstance.inst_list).code_gen()
        h.write(t_code)


def main(wrk_dir="./"):
    xml_list = []
    for roots, dirs, files in os.walk("./xml"):
        for f in files:
            xml_list.append(os.path.join(roots, f))
    root_list = [Root.Root(ET.parse(x).getroot()) for x in xml_list]
    gen_mc_header(root_list)


if __name__ == "__main__":
    main()
