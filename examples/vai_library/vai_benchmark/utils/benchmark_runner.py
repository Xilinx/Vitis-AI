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
import csv
import json
import re
import logging
from utils import utilities
from utils import summary

class BaseRunner:
    def __init__(self, csv_file_path, args):
        self.args = args
        self.models_with_row = self.load_csv(csv_file_path)
        self.example_path = '' if not args.example_path else args.example_path

    @staticmethod
    def load_csv(csv_file_path):
        models_with_row = {}
        try:
            with open(csv_file_path, 'r+') as csvfile:
                dict_reader = csv.DictReader(csvfile)
                for row in dict_reader:
                    models_with_row[row.get('ModelName')] = row
            if not models_with_row:
                logging.error('Not found any model in csv')
        except Exception as e:
            logging.error(e)
        finally:
            return models_with_row

    @staticmethod
    def xmodel_exists(models_path, xmodels, xmodel_extension):
        for xmodel in xmodels:
            if not xmodel:
                continue
            xmodel_path = os.path.join(models_path, xmodel, f'{xmodel}.{xmodel_extension}')
            if not utilities.file_exists(xmodel_path):
                logging.error(f'The xmodel is not installed: {xmodel_path}')
                return False
        else:
            return True

    @staticmethod
    def image_exists(images_path_list):
        for image_file_path in images_path_list:
            if not utilities.file_exists(image_file_path):
                logging.error(f'The testing image is not installed: {image_file_path}')
                return False
        else:
            return True

    @staticmethod
    def executabl_file_exists(executabl_file_path):
        if not utilities.file_exists(executabl_file_path):
            logging.error(f'The test file is not installed: {executabl_file_path}')
            return False
        return True

    def get_run_path(self, samples_dir, category):
        if self.example_path:
            run_path = os.path.join(self.example_path, samples_dir, category)
            return run_path

        first_run_path = os.path.join(utilities.FIRST_EXAMPLE_PATH, samples_dir, category)
        if os.path.exists(first_run_path):
            return first_run_path

        second_run_path = os.path.join(utilities.SECOND_EXAMPLE_PATH, samples_dir, category)
        if os.path.exists(second_run_path):
            return  second_run_path

    @staticmethod
    def wrap_model_env(cmd, model_env):
        wraped_cmd = cmd if not model_env else f'env {model_env} {cmd}'
        return wraped_cmd


class FunctionalityRunner(BaseRunner):
    # class for run functionality
    def generate_cmd(self, model_name, model_row):
        category = model_row.get('Category', '')
        executable_file = model_row.get('ExecutableFile', '')
        imagefile = model_row.get('ImageFile', '')
        xmodellist = model_row.get('XmodelList', '').strip()
        framework = model_row.get('FrameWork', '')
        model_env = model_row.get('ModelEnv', '')
        specified_cmd = model_row.get('SpecifiedCmd', '')
        xmodel_extension = 'xmodel' if framework != 'onnx' else 'onnx'
        samples_dir = 'samples' if framework != 'onnx' else 'samples_onnx'
        run_path = self.get_run_path(samples_dir, category)

        # get and check xmodel, testing image, and testing executable file
        images_list = [imagefile, ] if len(imagefile.split(' ')) == 1 else imagefile.split(' ')
        images_path_list = [f'{run_path}/{image_file}' for image_file in images_list]
        testing_image = ' '.join(images_path_list)
        executable_file_path = f'{run_path}/{executable_file}'
        models_path = utilities.DEFAULT_MODELS_PATH if not self.args.models_path else self.args.models_path
        xmodellist = [model_name, ] if not xmodellist else xmodellist.split(' ')

        if not self.xmodel_exists(models_path, xmodellist, xmodel_extension) \
                or not self.executabl_file_exists(executable_file_path) \
                or not self.image_exists(images_path_list):
            return run_path, None

        #  if specified cmd in csv, use the cmd directly
        if specified_cmd:
            return run_path, self.wrap_model_env(specified_cmd, model_env)

        # this is most model's test method, no specified models_path and no specified example path
        testing_xmodel_str = ' '.join(xmodellist)
        # if it's onnx sample or specified model path in parameter, needs full xmodel path in cmd
        if framework == 'onnx' or self.args.models_path:
            xmodel_paths = [os.path.join(models_path, xmodel, f'{xmodel}.{xmodel_extension}') for xmodel in xmodellist]
            testing_xmodel_str = ' '.join(xmodel_paths)

        cmd = f'{executable_file_path} {testing_xmodel_str} {testing_image}'
        return run_path, self.wrap_model_env(cmd, model_env)

    def execute(self, model_name):
        model_row = self.models_with_row.get(model_name, {})
        logging.info(f'Running Functionality: {model_name}')
        run_path, cmd = self.generate_cmd(model_name, model_row)
        if not cmd:
            return
        ret = utilities.execute_cmd(cmd, run_path=run_path)
        if ret.returncode == 0:
            stdout_str = '\n'.join([x for x in ret.stdout.decode("utf-8").strip().split('\n')
                                    if all([keyword not in x for keyword in
                                            ('XAIEFAL', 'Logging before InitGoogleLogging')])])
            stderr_str = '\n'.join([x for x in ret.stderr.decode("utf-8").strip().split('\n')
                                    if all([keyword not in x for keyword in
                                            ('XAIEFAL', 'Logging before InitGoogleLogging')])])
            if not self.args.summary:
                print(stdout_str + stderr_str)
        else:
            logging.error('Functionality cmd run failed!')


class E2EBenchmarkRunner(BaseRunner):
    # class for run E2E benchmark
    def generate_cmd(self, model_name, model_row, thread):
        category = model_row.get('Category', '')
        executabl_file = model_row.get('ExecutableFile', '')
        listfile = model_row.get('ListFile', '')
        xmodellist = model_row.get('XmodelList', '').strip()
        framework = model_row.get('FrameWork', '')
        model_env = model_row.get('ModelEnv', '')
        specified_cmd = model_row.get('SpecifiedCmd', '')
        xmodel_extension = 'xmodel' if framework != 'onnx' else 'onnx'
        samples_dir = 'samples' if framework != 'onnx' else 'samples_onnx'
        run_path = self.get_run_path(samples_dir, category)

        # get and check xmodel , testing images list, and testing executabl file
        executabl_file_path = f'{run_path}/{executabl_file}'
        models_path = utilities.DEFAULT_MODELS_PATH if not self.args.models_path else self.args.models_path
        xmodellist = [model_name, ] if not xmodellist else xmodellist.split(' ')
        listfile = f'test_performance_{category}.list' if not listfile else listfile

        if not self.xmodel_exists(models_path, xmodellist, xmodel_extension) \
                or not self.executabl_file_exists(executabl_file_path.strip().split(' ')[0]) \
                or not self.executabl_file_exists(f'{run_path}/{listfile}'):
            return run_path, None

        # if specified cmd in csv, use the cmd directly
        if specified_cmd:
            return run_path, self.wrap_model_env(specified_cmd, model_env)

        testing_xmodel_str = ' '.join(xmodellist)
        if framework == 'onnx' or self.args.models_path:
            xmodel_paths = [os.path.join(models_path, xmodel, f'{xmodel}.{xmodel_extension}') for xmodel in xmodellist]
            testing_xmodel_str = ' '.join(xmodel_paths)

        cmd = f'{executabl_file_path}  {testing_xmodel_str} -s {self.args.seconds} -t {thread} {listfile}'
        return run_path, self.wrap_model_env(cmd, model_env)

    def execute(self, model_name, threads=None):
        if model_name not in self.models_with_row.keys():
            logging.warning(f'Model {model_name} has no E2E performance testing yet.')
            return
        model_row = self.models_with_row.get(model_name, {})
        # threads = model_row.get('Threads', '').strip().split(' ')
        if not threads:
            threads = [1, 3, 4]
        logging.info(f'Running E2E benchmark: {model_name}')
        results = {}
        for thread in threads:
            run_path, cmd = self.generate_cmd(model_name, model_row, thread)
            if not cmd:
                continue
            ret = utilities.execute_cmd(cmd, run_path=run_path)
            if not ret.returncode == 0:
                continue
            logging.debug(ret.stdout.decode("utf-8").strip())
            lines = [x for x in ret.stdout.decode("utf-8").strip().split('\n') if 'XAIEFAL' not in x]
            for line in lines:
                datas = line.strip().split('=')
                if len(datas) != 2:
                    logging.error('Run failed!', ret.stdout.decode("utf-8").strip())
                    continue
                if datas[0].strip() == 'FPS':
                    results[str(thread)] = datas[1]

        if not results:
            return

        if self.args.summary:
            utilities.save_dict_to_json(os.path.join(utilities.PATH_TO_SAVE_LOG, 'e2e', model_name + '.json'), results)
        else:
            e2e_title = ['Model Name', 'Thread', 'E2E(FPS)']
            summary.Summary().pretty_print(e2e_title, {model_name: results, })


class DPUBenchmarkRunner(BaseRunner):
    # class for run only DPU benchmark
    @staticmethod
    def get_dpu_subgraph_index(xmodel_path):
        cmd = 'xdputil xmodel -l %s' % xmodel_path
        rc = utilities.execute_cmd(cmd)

        xmodel_meta = json.loads(rc.stdout.decode("utf-8"))
        dpu_subgraphs = [x for x in xmodel_meta.get("subgraphs", []) if x.get('device', '') == 'DPU']
        if len(dpu_subgraphs) != 1:
            logging.warning('get dpu subgraph index failed')
            return -1
        dpu_subgraph = dpu_subgraphs[0]
        dpu_subgraph_index = dpu_subgraph.get('index')

        return dpu_subgraph_index

    def test_xdputil_benchmark(self, xmodel_path, dpu_subgraph_index='1', threads=None):
        if not threads:
            threads = [1, 3, 4]
        results = {}
        utilities.set_env({'SLEEP_MS': str(int(self.args.seconds) * 1000), })
        for thread in threads:
            run_cmd = 'xdputil benchmark %s -i %s %s' % (xmodel_path, dpu_subgraph_index, thread)
            rc = utilities.execute_cmd(run_cmd)
            output = rc.stderr.decode("utf-8")
            if not output:
                continue
            lines = output.split('\n')
            for line in lines:
                m = re.match(r'.*?FPS=\s(\d+\.\d+)\s.*?', line)
                if m:
                    model_fps = m.group(1)
                    results[str(thread)] = model_fps
                    logging.debug('DPU benchmark with %s thread: %s img/s' % (thread, model_fps))
                    break
            else:
                logging.warning(f"Thread {thread}: Not found fps result!")
        return results

    def execute(self, model_name, threads=None):
        models_path = utilities.DEFAULT_MODELS_PATH if not self.args.models_path else self.args.models_path
        model_row = self.models_with_row.get(model_name, {})
        xmodellist = model_row.get('XmodelList', '').strip()
        xmodellist = [model_name, ] if not xmodellist else xmodellist.split(' ')
        framework = model_row.get('FrameWork', '')
        if framework == 'onnx':
            logging.warning('ONNX model not support dpu benchmark yet.')
            return
        if not self.xmodel_exists(models_path, xmodellist, 'xmodel'):
            return

        logging.info(f'Running DPU benchmark: {model_name}')
        for xmodel_name in xmodellist:
            xmodel_path = os.path.join(models_path, xmodel_name, f'{xmodel_name}.xmodel')
            dpu_subgraph_index = self.get_dpu_subgraph_index(xmodel_path)
            results = self.test_xdputil_benchmark(xmodel_path, dpu_subgraph_index, threads)

            if not results:
                return

            if self.args.summary:
                utilities.save_dict_to_json(os.path.join(utilities.PATH_TO_SAVE_LOG, 'dpu', xmodel_name + '.json'), results)
            else:
                dpu_title = ['Model Name', 'Thread', 'DPU(FPS)']
                summary.Summary().pretty_print(dpu_title, {xmodel_name: results, })
