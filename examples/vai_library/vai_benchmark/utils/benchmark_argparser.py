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

import argparse


class BenchmarkArgparser:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='')
        query_group = self.parser.add_mutually_exclusive_group()
        query_group.add_argument('--query', dest='query', action='store_true',
                                 help='Query the target, show key elements.')
        query_group.add_argument('--query_all', dest='query_all', action='store_true',
                                 help='Query the target, show all information.')

        mode_group = self.parser.add_mutually_exclusive_group()
        mode_group.add_argument('-func', '--functionality', dest='functionality',
                                help="Test model's functionality, which needs testing image.", action='store_true')
        mode_group.add_argument('-e2e', '--e2e_benchmark', dest='e2e_benchmark',
                                help="Test model's E2E benchmark, which needs testing image.", action='store_true')
        mode_group.add_argument('-dpu', '--dpu_benchmark', dest='dpu_benchmark',
                                help="Test model's DPU benchmark, only xmodel is necessary.", action='store_true')
        mode_group.add_argument('-full', '--full', dest='full', action='store_true',
                                help="Full test of model, functionality and DPU E2E benchmark")

        testing_model_group = self.parser.add_mutually_exclusive_group()
        testing_model_group.add_argument('-m', '--model_name', dest='model_name',
                                     help="Only run specified models, use ',' to separate multiple models.", type=str)
        testing_model_group.add_argument('-g', '--group', dest='group', default='KeyModel',
                                         choices=['KeyModel', 'all', 'typical'],
                                         help='Run models in specified group, default is KeyModel.')

        self.parser.add_argument('--debug', dest='debug', help='Enable debug logs.', action='store_true')
        self.parser.add_argument('--summary', dest='summary', action='store_true',
                                 help='Integrate DPU and E2E benchmark results in one table.')

        self.parser.add_argument('-models_path', dest='models_path', help='The path of saved models.', type=str)
        self.parser.add_argument('-t', '--threads', dest='threads', help='Assigned benchmark threads.', type=str)
        self.parser.add_argument('-s', '--seconds', dest='seconds', default='60',
                                 help='The number of running seconds for every model, default is 60s.', type=str)
        self.parser.add_argument('-example_path', dest='example_path',
                                 help='The path of saved vitis_ai_library example.', type=str)

    def make_args(self):
        return self.parser.parse_args()
