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

from xdputil_component import *
import xdputil_component


def main():
    import argparse

    parser = argparse.ArgumentParser(description="xilinx tools")
    parser.add_argument("-v",
                        "--version",
                        action="version",
                        version="%(prog)s 1.0")
    subparsers = parser.add_subparsers(title="sub command ",
                                       description="xmodel tools",
                                       help="sub-command help")
    for i in xdputil_component.__all__:
        m = getattr(xdputil_component, i)
        m.help(subparsers)
    args = parser.parse_args()

    args.func(args)


if __name__ == "__main__":
    main()
