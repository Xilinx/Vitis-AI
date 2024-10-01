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
#

with section("parse"):
    additional_commands = {
        'my_pybind11_add_module': {
            'kwargs': {
                'MODULE_NAME': 1,
                'PACKAGE_NAME': 1
            },
            'pargs': {
                'flags': [],
                'nargs': '1+'
            }
        },
        'vai_add_library': {
            'kwargs': {
                'INCLUDE_DIR': 1,
                'NAME': 1,
                'PRIVATE_REQUIRE': '+',
                'PUBLIC_HEADER': '+',
                'PUBLIC_REQUIRE': '+',
                'SRCS': '+',
                'SRC_DIR': 1,
                'TESTS': '+',
                'TEST_DIR': 1
            },
            'pargs': {
                'flags': ['STATIC', 'SHARED', 'MODULE', 'NOT_GLOB_SRC'],
                'nargs': '*'
            }
        },
        'vai_add_test': {
            'kwargs': {
                'ENABLE_IF': 1,
                'REQUIRE': '+'
            },
            'pargs': {
                'flags': [],
                'nargs': '1+'
            }
        },
        'vai_add_sample': {
            'kwargs': {
                'REQUIRE': '+',
                'SRCS': '+'
            },
            'pargs': {
                'flags': [],
                'nargs': '1+'
            }
        },
        "vai_overview_add_accuracy": {
            "kwargs": {
                "REQUIRE": "+",
            },
            "pargs": {"flags": [], "nargs": "1+"},
        },
    }
