"""
 Copyright 2019 Xilinx Inc.

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

from pathlib import Path

from xir import Graph

from xnnc import cli, runner


def serialize(graph: Graph, out_dir: Path):
    # get graph name
    name = graph.get_name()

    # * serialize graph
    fxmodel = out_dir / (name + ".xmodel")
    serialize_as_file(graph, fxmodel)


def serialize_as_file(graph: Graph, filename: Path):
    if not filename.parent.exists():
        filename.parent.mkdir(parents=True)

    print(f"[INFO] dump xmodel ...", end="\r")
    graph.serialize(str(filename))
    assert filename.exists() == True
    print(f"[INFO] dump xmodel: {filename.absolute()}")


# * entry point of xnnc app
def main(args=None):
    # parse CLI arguments
    args = cli.parse_args(args)
    print(f"[INFO] {args}")

    runner.normal_run(args)
