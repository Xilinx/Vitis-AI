

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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("SigmoidTableLookup", &SigmoidTableLookup, "SigmoidTableLookup");
    m.def("SigmoidSimulation", &SigmoidSimulation, "SigmoidSimulation");
    m.def("TanhTableLookup",    &TanhTableLookup,    "TanhTableLookup");
    m.def("TanhSimulation",    &TanhSimulation,    "TanhSimulation");
    m.def("Scale",              &Scale,              "Scale");
    m.def("FixNeuronV2",        &FixNeuronV2,        "FixNeuronV2");
    m.def("DiffsFixPos",        &DiffsFixPos,        "DiffsFixPos");
}

/*
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("SigmoidTableLookup", &SigmoidTableLookup, "SigmoidTableLookup(cuda)");
    m.def("TanhTableLookup",    &TanhTableLookup,    "TanhTableLookup(cuda)");
    m.def("Scale",              &Scale,              "Scale(cuda)");
    m.def("FixNeuronV2",        &FixNeuronV2,        "FixNeuronV2(cuda)");
    m.def("DiffsFixPos",        &DiffsFixPos,        "DiffsFixPos(cuda)");
}
*/
