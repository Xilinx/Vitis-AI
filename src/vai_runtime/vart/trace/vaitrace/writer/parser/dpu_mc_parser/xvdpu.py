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

from .dpuMcParserBase import *


class xvdpu_InstParser(dpuMcParserBase):
    def __init__(self, name):
        super().__init__(name)
        local_dir = os.path.dirname(__file__)
        try:
            self.mc_prser_cdll = CDLL("%s/%s.so" % (local_dir, name.lower()))
            self.process_cpp = self.mc_prser_cdll.process
        except:
            self.mc_prser_cdll = None
            self.process_cpp = None

    def process(self, mc, _debug=False):
        if self.process_cpp != None:
            debug_c = c_bool(_debug)
            mc_c = c_char_p(mc)
            mc_len_c = c_uint(len(mc))

            load_img_size_c = c_uint(0)
            load_para_size_c = c_uint(0)
            save_size_c = c_uint(0)

            # void process(uint8_t *mc, bool debug, uint32_t *load_img_size, uint32_t *load_para_size, uint32_t *save_size)
            self.process_cpp(mc_c, mc_len_c, debug_c, pointer(
                load_img_size_c), pointer(load_para_size_c), pointer(save_size_c))

            self.data["load_img_size"] = load_img_size_c.value
            self.data["load_para_size"] = load_para_size_c.value
            self.data["save_size"] = save_size_c.value
        else:
            pass


register(xvdpu_InstParser("DPUCVDX8G_ISA1"))
register(xvdpu_InstParser("DPUCVDX8G_ISA2"))
register(xvdpu_InstParser("DPUCVDX8G_ISA3"))
register(xvdpu_InstParser("DPUCV2DX8G_ISA0"))
register(xvdpu_InstParser("DPUCV2DX8G_ISA1"))
