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
import subprocess
import sys
import json
import logging


PATH_TO_SAVE_LOG = '/tmp/.vai_benchmark'
DEFAULT_MODELS_PATH = '/usr/share/vitis_ai_library/models'
FIRST_EXAMPLE_PATH = '/home/root/Vitis-AI/examples/vai_library'
SECOND_EXAMPLE_PATH = '/usr/share/vitis_ai_library'

def execute_cmd(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, capture_output=False, run_path=''):
    pwd = os.getcwd()
    try:
        if run_path:
            os.chdir(run_path)
        logging.debug('Execute cmd: %s' % cmd)
        if capture_output:
            ret = subprocess.run(cmd, shell=shell, capture_output=capture_output)
        else:
            ret = subprocess.run(cmd, shell=shell, stdout=stdout, stderr=stderr)
        if ret.returncode != 0:
            logging.error('Returncode is wrong.')
            logging.error(ret.stderr.decode("utf-8"))
        return ret
    except Exception as e:
        logging.error(e)
    finally:
        if run_path:
            os.chdir(pwd)


def file_exists(file_path):
    return True if os.path.exists(file_path) else False


def makedirs(dir_path):
    try:
        os.makedirs(dir_path)
    except Exception as e:
        logging.error(e)


def rmdir(dir_path):
    try:
        import shutil
        shutil.rmtree(dir_path)
    except Exception as e:
        logging.error(e)


def read_json_to_dict(file_path):
    try:
        if not os.path.exists(file_path):
            logging.error('File NOT EXISTS: %s' % file_path)
            return
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logging.error(e)


def save_dict_to_json(result_file, results):
    try:
        logging.debug('Save results to %s' % result_file)
        with open(result_file, 'w') as f:
            json.dump(results, f, indent = 4)
    except Exception as e:
        logging.error(e)


def read_file(file_path):
    try:
        if not os.path.exists(file_path):
            logging.error('File NOT EXISTS: %s' % file_path)
            return
        with open(file_path, 'r') as f:
            return f.read()
    except Exception as e:
        logging.error(e)


def set_env(env_dict):
    for key, value in env_dict.items():
        logging.debug(f'set env {key}={value}')
        os.environ[key] = value
