#!/usr/bin/env python
#-*- coding:utf-8 -*-

'''
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
'''

from setuptools import setup, find_packages
setup (
  name = "Edge Vitis AI",

  version="1.2",

  packages = find_packages(),

  zip_safe = False,

  description = "DNNDK APIs.",
  long_description= "",

  author = "Xilinx",

  author_email = "",

  license = "Xilinx All Rights Reserved",
  
  keywords = ("dnndk"),

  platforms = "Independant",

  url = "",       
        
)
