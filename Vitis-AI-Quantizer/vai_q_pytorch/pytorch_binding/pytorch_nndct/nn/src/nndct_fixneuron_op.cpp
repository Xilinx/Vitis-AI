

/* 
# (c) Copyright 2016 – 2019 Xilinx, Inc. All rights reserved. 
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
# INFRINGEMENT, OR FITNESS FOR ANY PARTICULAR PURPOSE;
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
# PART OF THIS FILE AT ALL TIMES
*/

#include <ATen/ATen.h>
#include "c10/util/ArrayRef.h"
#include "../include/nndct_fixneuron_op.h"
#include  "../../../../include/nndct_fix_kernels.h"

template <typename Dtype>
void _FixNeuronV2(Tensor Tinput,
                 Tensor Toutput, 
                 int valmax, 
                 Dtype valamp, 
                 int method){

    auto input = Tinput.data<Dtype>();
    auto output = Toutput.data<Dtype>();
    int64_t num_ele = Tinput.numel();

    cuda_fix_neuron_v2(num_ele, 
                       input, 
                       output,
                       valmax, 
                       valamp, 
                       1, //keep_scale
                       method);
}

void FixNeuronV2(Tensor Tinput,
                 Tensor Toutput, 
                 int valmax, 
                 float valamp, 
                 int method){
  if (Tinput.dtype() == at::kFloat)
    _FixNeuronV2<float>(Tinput,
                        Toutput, 
                        valmax, 
                        valamp, 
                        method);
  else if (Tinput.dtype() == at::kDouble)
    _FixNeuronV2<double>(Tinput,
                         Toutput, 
                         valmax, 
                         valamp, 
                         method);
}

