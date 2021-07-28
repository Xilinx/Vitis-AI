

#
# Copyright 2019 Xilinx Inc.
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

from .option_def import Option

####################################
# Add Option list here
####################################

class NndctOption(object):
  nndct_help = Option(name="help", dtype=bool, default=False, action="store_true", 
                      help="list all api usage description")

  nndct_quant_off = Option(name="quant_off", dtype=bool, default=False, action="store_true", 
                           help="disable quantization flow")

  nndct_option_list = Option(name="option_list", dtype=bool, default=False, action="store_true", 
                             help="list all the options in nndct")

  nndct_parse_debug = Option(name="parse_debug", dtype=int, default=0, 
                             help="logging graph, 1: torch raw graph, 2: nndct graph 3: nndct quant graph")

  nndct_logging_level = Option(name="logging_level", dtype=int, default=0, help="logging level")
  
  nndct_quant_mode = Option(name="quant_mode", dtype=int, default=0, 
                            help="quant mode, 1:calibration, 2:quantization")
  
  nndct_dump_float_format = Option(name="dump_float_format", dtype=int, default=0, 
                            help="deploy check data format, 0: bin, 1: txt")
  
  nndct_record_slow_mode = Option(name="record_slow_mode", dtype=bool, default=False, action="store_true", 
                                  help="record outputs every iteration")
  
  nndct_deploy_check = Option(name="deploy_check", dtype=bool, default=False, action="store_true", 
                              help="dump deploy data")
  
  nndct_quant_opt = Option(name="quant_opt", dtype=int, default=3, help="quant opt level")
  
  nndct_relu6_replace = Option(name="relu6_replace", dtype=str, default='relu', help="relu6 replace operator")

  nndct_equalization = Option(name="equalization", dtype=bool, default=True, action="store_true", 
                              help="enable weights equalization")
  
  nndct_wes = Option(name="weights_equalizing_shift", dtype=bool, default=False, action="store_true",
                     help="enable weights equalizing shift")
  
  nndct_wes_in_cle = Option(name="weights_equalizing_shift in cle", dtype=bool, default=False, action="store_true",
                     help="enable weights equalizing shift in cle")

  nndct_param_corr = Option(name="param_corr", dtype=bool, default=True, action="store_true", 
                            help="enable parameter correction")

  nndct_param_corr_rate = Option(name="param_corr_rate", dtype=float, default=0.05, help="parameter correction rate")

  nndct_cv_app = Option(name="cv_app", dtype=bool, default=True, action="store_true", help="cv application")

  nndct_finetune_lr_factor = Option(name="finetune_lr_factor", dtype=float, default=0.01, help="finetune learning rate factor")

  nndct_stat = Option(name="stat", dtype=int, default=0, help="quantizer statistic level")
  
