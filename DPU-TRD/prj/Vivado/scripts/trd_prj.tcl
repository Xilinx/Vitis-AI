# (c) Copyright 2018-2018 Xilinx, Inc. All rights reserved.
#
# This file contains confidential and proprietary information
# of Xilinx, Inc. and is protected under U.S. and
# international copyright and other intellectual property
# laws.
#
# DISCLAIMER
# This disclaimer is not a license and does not grant any
# rights to the materials distributed herewith. Except as
# otherwise provided in a valid license issued to you by
# Xilinx, and to the maximum extent permitted by applicable
# law: (1) THESE MATERIALS ARE MADE AVAILABLE "AS IS" AND
# WITH ALL FAULTS, AND XILINX HEREBY DISCLAIMS ALL WARRANTIES
# AND CONDITIONS, EXPRESS, IMPLIED, OR STATUTORY, INCLUDING
# BUT NOT LIMITED TO WARRANTIES OF MERCHANTABILITY, NON-
# INFRINGEMENT, OR FITNESS FOR ANY PARTICULAR PURPOSE; and
# (2) Xilinx shall not be liable (whether in contract or tort,
# including negligence, or under any other theory of
# liability) for any loss or damage of any kind or nature
# related to, arising under or in connection with these
# materials, including for any direct, or any indirect,
# special, incidental, or consequential loss or damage
# (including loss of data, profits, goodwill, or any type of
# loss or damage suffered as a result of any action brought
# by a third party) even if such damage or loss was
# reasonably foreseeable or Xilinx had been advised of the
# possibility of the same.
#
# CRITICAL APPLICATIONS
# Xilinx products are not designed or intended to be fail-
# safe, or for use in any application requiring fail-safe
# performance, such as life-support or safety devices or
# systems, Class III medical devices, nuclear facilities,
# applications related to the deployment of airbags, or any
# other applications that could lead to death, personal
# injury, or severe property or environmental damage
# (individually and collectively, "Critical
# Applications"). Customer assumes the sole risk and
# liability of any use of Xilinx products in Critical
# Applications, subject only to applicable laws and
# regulations governing limitations on product liability.
#
# THIS COPYRIGHT NOTICE AND DISCLAIMER MUST BE RETAINED AS
# PART OF THIS FILE AT ALL TIMES.
#

#****************************************************************
# set global dict_prj
#****************************************************************
set dict_prj  {}

#****************************************************************
# set project
#****************************************************************
dict set dict_prj dict_sys prj_name                  {zcu102}
dict set dict_prj dict_sys prj_part                  {xczu9eg-ffvb1156-2-i}
dict set dict_prj dict_sys prj_board                 {zcu102v100}

#****************************************************************
# set bd
#   for bd_ooc: None for global, Hierarchical for ooc per IP
#****************************************************************
dict set dict_prj dict_sys bd_name                   top
dict set dict_prj dict_sys bd_ooc                    None

#****************************************************************
# set param
#****************************************************************
dict set dict_prj dict_param  DPU_CLK_MHz            {325}
dict set dict_prj dict_param  DPU_NUM                {2}
dict set dict_prj dict_param  DPU_ARCH               {4096}
dict set dict_prj dict_param  DPU_RAM_USAGE          {low} 
dict set dict_prj dict_param  DPU_CHN_AUG_ENA        {1}
dict set dict_prj dict_param  DPU_DWCV_ENA           {1}
dict set dict_prj dict_param  DPU_AVG_POOL_ENA       {1}
dict set dict_prj dict_param  DPU_CONV_RELU_TYPE     {3}
dict set dict_prj dict_param  DPU_SFM_NUM            {1}
dict set dict_prj dict_param  DPU_DSP48_USAGE        {high}
dict set dict_prj dict_param  DPU_URAM_PER_DPU       {0}

dict set dict_prj dict_param  REG_CLK_MHz            {100}
dict set dict_prj dict_param  HP_CLK_MHz             {334}
dict set dict_prj dict_param  CN_STGY                {flat}
dict set dict_prj dict_param  DPU_IP_FOLDER          {dpu}
dict set dict_prj dict_param  DPU_SAXICLK_INDPD      {1}
dict set dict_prj dict_param  DPU_CLK_GATING_ENA     {1}
dict set dict_prj dict_param  DPU_DSP48_MAX_CASC_LEN {4}
dict set dict_prj dict_param  DPU_TIMESTAMP_ENA      {1}

#****************************************************************
# source tcl
#****************************************************************
dict set dict_prj dict_sys work_dir                  [file dirname [file normalize [info script]]]
dict set dict_prj dict_sys tcl_base_dir              [dict get $dict_prj dict_sys work_dir]/base
source -notrace                                      [dict get $dict_prj dict_sys tcl_base_dir]/trd_bd.tcl
#****************************************************************
# run flow
#****************************************************************
lib_flow

# 41d051e7f991c9dacbe8d368b85f57380b272bd2d7dc26e6c472a2d06e70908e
