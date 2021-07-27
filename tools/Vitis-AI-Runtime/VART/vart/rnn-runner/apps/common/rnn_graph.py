"""
Copyright 2021 Xilinx Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import xir


def create_rnn_graph(model_dir: str, model_name: str,
                     num_sequences: int, input_seq_dim: int,
                     output_seq_dim: int, device_name: str,
                     device_core_id: int):
    fakegraph = xir.Graph("lstm")
    rs = fakegraph.get_root_subgraph()

    rs.set_attr("runner", {"run": "libvart-rnn-runner.so"})
    rs.set_attr("device_name", device_name)
    rs.set_attr("device_core_id", device_core_id)
    rs.set_attr("model_dir", model_dir)
    rs.set_attr("model_name", model_name)
    rs.set_attr("num_sequences", num_sequences)
    rs.set_attr("input_seq_dim", input_seq_dim)
    rs.set_attr("output_seq_dim", output_seq_dim)
    return fakegraph
