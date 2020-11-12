
# This code is based on: https://github.com/nutonomy/second.pytorch.git
# 
# MIT License
# Copyright (c) 2018 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import pickle

import second.core.preprocess as prep
from second.builder import preprocess_builder
from second.core.preprocess import DataBasePreprocessor
from second.core.sample_ops import DataBaseSamplerV2


def build(sampler_config):
    cfg = sampler_config
    groups = list(cfg.sample_groups)
    prepors = [
        preprocess_builder.build_db_preprocess(c)
        for c in cfg.database_prep_steps
    ]
    db_prepor = DataBasePreprocessor(prepors)
    rate = cfg.rate
    grot_range = cfg.global_random_rotation_range_per_object
    groups = [dict(g.name_to_max_num) for g in groups]
    info_path = cfg.database_info_path
    with open(info_path, 'rb') as f:
        db_infos = pickle.load(f)
    grot_range = list(grot_range)
    if len(grot_range) == 0:
        grot_range = None
    sampler = DataBaseSamplerV2(db_infos, groups, db_prepor, rate, grot_range)
    return sampler
