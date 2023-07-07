

/*
* Copyright 2019 Xilinx Inc.
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/


#include <torch/extension.h>
#include "../../include/nndct_fixneuron_op.h"
#include "../../include/nndct_diffs_op.h"
#include "../../include/nndct_math.h"
#include "../../include/bfp.h"

// For registration in torch script
#ifdef TORCH_LIBRARY
TORCH_LIBRARY(vai, m) {
  m.def("fix_neuron", fix_neuron);
  m.def("diffs_fix_pos", diffs_fix_pos);
  m.def("SigmoidTableLookup", SigmoidTableLookup);
  m.def("SigmoidSimulation", SigmoidSimulation);
  m.def("TanhTableLookup", TanhTableLookup);
  m.def("TanhSimulation", TanhSimulation);
  m.def("SoftmaxExpApproximate", SoftmaxExpApproximate);
  m.def("SoftmaxLOD", SoftmaxLOD);
  m.def("SoftmaxSimulationPart1", SoftmaxSimulationPart1);
  m.def("SoftmaxSimulationPart2", SoftmaxSimulationPart2);
  m.def("SigmoidTableLookupAIE2", SigmoidTableLookupAIE2);
  m.def("TanhTableLookupAIE2", TanhTableLookupAIE2);
  m.def("ExpApprAIE2", ExpApprAIE2);
  m.def("LogSoftmaxFastLn", LogSoftmaxFastLn);
  m.def("LogSoftmaxSub", LogSoftmaxSub);
   m.def("LayernormISqrt", LayernormISqrt);
  m.def("LayernormInvSqrt", LayernormInvSqrt);
  m.def("InverseAIE2", InverseAIE2);
  m.def("to_bfp", to_bfp);
  m.def("to_bfp_v2", to_bfp_v2);
  //m.def("to_bfp_v2(Tensor input, int bit_width, int block_size, Tensor output) -> Tensor");
}
#else
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("SigmoidTableLookup", &SigmoidTableLookup, "SigmoidTableLookup");
  m.def("SigmoidSimulation", &SigmoidSimulation, "SigmoidSimulation");
  m.def("TanhTableLookup", &TanhTableLookup, "TanhTableLookup");
  m.def("TanhSimulation", &TanhSimulation, "TanhSimulation");
  m.def("FixNeuronV2", &FixNeuronV2, "FixNeuronV2");
  m.def("DiffsFixPos", &DiffsFixPos, "DiffsFixPos");
  m.def("SoftmaxExpApproximate", &SoftmaxExpApproximate, "SoftmaxExpApproximate");
  m.def("SoftmaxLOD", &SoftmaxLOD, "SoftmaxLOD");
  m.def("SoftmaxSimulationPart1", &SoftmaxSimulationPart1, "SoftmaxSimulationPart1");
  m.def("SoftmaxSimulationPart2", &SoftmaxSimulationPart2, "SoftmaxSimulationPart2");
  m.def("InverseAIE2", &InverseAIE2, "InverseAIE2");
  m.def("to_bfp", &to_bfp, "to_bfp");
  m.def("to_bfp_v2", &to_bfp_v2, "to_bfp_v2");
  m.def("LayernormISqrt", &LayernormISqrt, "LayernormISqrt");
}
#endif
