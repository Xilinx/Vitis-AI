#!/usr/bin/env python
# coding=utf-8
"""
Copyright 2022-2023 Advanced Micro Devices Inc.

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

import json
import tools_extra_ops as tools


def main(args):
    print(
        json.dumps(
            tools.xdputil_status(), sort_keys=True, indent=4, separators=(",", ":")
        )
    )


def help(subparsers):
    parser = subparsers.add_parser("status", help="")

    parser.set_defaults(func=main)
