#
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
import unittest
import xir
import vart
import xir_extra_ops
import numpy as np
import simple_graph


class AllTest(unittest.TestCase):
    def test_plate_number(self):
        g = xir.Graph.deserialize("plate_num.xmodel")
        simple_graph.set_reference_input (g, "plate_number.golden/plate_data.bin")
        simple_graph.add_check_point (g, {("score1m", "download"):
                                          "plate_number.golden/score1m.bin"})
        simple_graph.normal_setting_for_graph (g)
        simple_graph.run_graph (g)
    def test_softmax(self):
        g = xir.Graph("test_softmax")
        data = np.asarray([[2.0, 2.0, 3.0, 4.0]], dtype=np.float32);
        data_op = g.create_const_op ("input", data)
        data_op.set_attr("subgraph_path", ["cpu1"])
        softmax_op = g.create_op("softmax",
                                 "softmax",
                                 attrs={"axis": -1,
                                        "subgraph_path": ["cpu1"]},
                                 input_ops= {"input": [data_op]});
        compare_op = g.create_op("compare",
                                 "compare",
                                 attrs={
                                     "from_file": "test/ref_results/softmax0.bin",
                                     "subgraph_path": ["cpu1"]
                                 },
                                 input_ops= {"input": [softmax_op]});
        simple_graph.create_simple_cpu_graph (g)
        simple_graph.run_graph (g)

if __name__ == '__main__':
    unittest.main()
