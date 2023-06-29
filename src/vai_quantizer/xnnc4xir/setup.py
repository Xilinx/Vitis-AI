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

from setuptools import setup, find_packages

with open("README.md", "r") as f:
    long_desc = f.read()

setup(
    name="xnnc",
    version="3.5.0",
    description="eXtensible Neural Network Converter",
    packages=find_packages(exclude=("test", "docs")),
    zip_safe=False,
    entry_points={"console_scripts": ["xnnc-run=xnnc.__main__:main"]},
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "Operating System :: OS Independent",
    ],
    include_package_data=True,
)
