#!/usr/bin/python3
# coding:utf-8

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


import os
import sys
import logging
from utils import utilities, benchmark_argparser, benchmark_runner, query, summary


def get_running_thread(cu_number, batch_number):
    # get the benchmark default run thread
    if cu_number > 1:
        default_thread = cu_number * 2
        logging.debug('Benchmark default running thread: %s' % default_thread)
        return default_thread

    default_thread = 6 if batch_number <= 6 else batch_number // 2
    logging.debug('Benchmark default running thread: %s' % default_thread)
    return default_thread


def main():
    # Set Parameters
    arg_parser = benchmark_argparser.BenchmarkArgparser()
    args = arg_parser.make_args()

    if args.debug:
        logging.basicConfig(level=logging.DEBUG, format="%(filename)s:%(lineno)d - %(levelname)s - %(message)s")
    else:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

    query_runner = query.TargetQuery(args)
    query_runner.query()
    if args.query:
        query_runner.show()
        sys.exit()
    if args.query_all:
        query_runner.show('all')
        sys.exit()

    # get the benchmark default run thread
    if args.threads:
        threads = args.threads.split(',')
    else:
        cu_number = query_runner.target_info.get('DPU Compute Unit Number', 1)
        batch_number = query_runner.target_info.get('DPU Batch Number', 1)
        logging.debug('DPU Compute Unit Number: %s' % cu_number)
        logging.debug('DPU Batch Number: %s' % batch_number)
        threads = [get_running_thread(cu_number, batch_number), ]

    # get csv path
    current_file_path = os.path.abspath(__file__)
    current_dir = os.path.split(current_file_path)[0]
    func_csv = os.path.join(current_dir, 'benchmark_csv', 'functionality.csv')
    perf_csv = os.path.join(current_dir, 'benchmark_csv', 'performance.csv')
    # create benchmark runners
    func_runner = benchmark_runner.FunctionalityRunner(func_csv, args)
    e2e_benchmark_runner = benchmark_runner.E2EBenchmarkRunner(perf_csv, args)
    dpu_benchmark_runner = benchmark_runner.DPUBenchmarkRunner(func_csv, args)

    # get the models list
    if args.model_name:
        running_models = args.model_name.split(',')
    elif args.group and args.group == 'all':
        running_models = list(func_runner.models_with_row.keys())
    elif args.group and args.group == 'typical':
        running_models = [model_name for model_name in func_runner.models_with_row.keys() if 'typical' in
                          func_runner.models_with_row.get(model_name, {}).get('Group', '').strip().split(',')]
    else:  # run default models in group KeyModel
        running_models = [model_name for model_name in func_runner.models_with_row.keys() if 'KeyModel' in
                          func_runner.models_with_row.get(model_name, {}).get('Group', '').strip().split(',')]

    if args.summary:
        if os.path.exists(utilities.PATH_TO_SAVE_LOG):
            logging.debug('rm %s' % utilities.PATH_TO_SAVE_LOG)
            utilities.rmdir(utilities.PATH_TO_SAVE_LOG)
        logging.debug('mkdir %s ' % os.path.join(utilities.PATH_TO_SAVE_LOG, 'dpu'))
        logging.debug('mkdir %s ' % os.path.join(utilities.PATH_TO_SAVE_LOG, 'e2e'))
        utilities.makedirs(os.path.join(utilities.PATH_TO_SAVE_LOG, 'dpu'))
        utilities.makedirs(os.path.join(utilities.PATH_TO_SAVE_LOG, 'e2e'))
    # run model on board
    for model_name in running_models:
        # logging.info(f'Testing model {model_name}:')
        if args.functionality or args.full:
            func_runner.execute(model_name)
        if args.dpu_benchmark or args.full:
            dpu_benchmark_runner.execute(model_name, threads)
        if args.e2e_benchmark or args.full:
            e2e_benchmark_runner.execute(model_name, threads)

    if args.summary:
        summary_runner = summary.Summary()
        summary_runner.show()


if __name__ == "__main__":
    main()
