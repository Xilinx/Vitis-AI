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
import json
import re
import logging

from . import utilities


class Summary:
    def show(self):
        dpu_results = self.get_results_dict('dpu')
        e2e_results = self.get_results_dict('e2e')

        if dpu_results:
            dpu_title = ['Model Name', 'Thread', 'DPU(FPS)']
            logging.info('DPU Benchmark Summary:')
            self.pretty_print(dpu_title, dpu_results)
        if e2e_results:
            logging.info('E2E Benchmark Summary:')
            e2e_title = ['Model Name', 'Thread', 'E2E(FPS)']
            self.pretty_print(e2e_title, e2e_results)

    @staticmethod
    def get_results_dict(type):
        # get dpu or e2e result from default utilities.PATH_TO_SAVE_LOG
        results_dict = {}
        log_list = [x for x in os.listdir(os.path.join(utilities.PATH_TO_SAVE_LOG, type))]
        for log_file in log_list:
            model_name = os.path.splitext(log_file)[0]
            log_file_path = os.path.join(utilities.PATH_TO_SAVE_LOG, type, log_file)
            results = utilities.read_json_to_dict(log_file_path)
            results_dict[model_name] = results
        return results_dict

    @staticmethod
    def pretty_print(title, results_dict):
        thread_len = 6  # len('Thread')
        fps_len = 10  # len('12345.78')
        max_model_name_len = max([len(x) for x in results_dict.keys()])
        if max_model_name_len > len(title[0]):
            model_name_colume_len = max_model_name_len
            model_name_len_diff = max_model_name_len - len(title[0])
            title[0] = title[0] + ' ' * model_name_len_diff
        else:
            model_name_colume_len = len(title[0])

        table_width = model_name_colume_len + thread_len + fps_len + len(' | ') * 2
        pretty_table = list()
        pretty_table.append('=' * table_width)
        pretty_table.append(' | '.join(title))

        for model_name, results in results_dict.items():
            model_len_diff = model_name_colume_len - len(model_name)
            model_name_str = model_name + ' ' * model_len_diff

            separator = "-+-".join(['-' * n for n in (model_name_colume_len, thread_len, fps_len)])
            pretty_table.append(separator)
            for index, (thread, fps_result) in enumerate(results.items()):
                thread_str = str(thread) + ' ' * (thread_len - len(thread))
                fps_str = str(fps_result) + ' ' * (fps_len - len(fps_result))
                model_name_str = model_name_str if index == 0 else ' ' * max_model_name_len
                row_str = ' | '.join((model_name_str, thread_str, fps_str))
                pretty_table.append(row_str)

        pretty_table.append('=' * table_width)

        print('\n'.join(pretty_table))


if __name__ == "__main__":
    import utilities
    summary = Summary()
    summary.show()
