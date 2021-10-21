/*
 * Copyright 2021 Xilinx, Inc.
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
// This file holds the body for the test harness of the matrix mult graph class

#include <stdio.h>
#include "test.hpp"

xf::dsp::aie::testcase::test_graph matMult;

#ifdef USING_UUT
#ifdef USING_PL_MOVER
// need explicit port name, so we can connect in hw.
PLIO* in1 = new PLIO("DataIn1", adf::plio_32_bits, QUOTE(INPUT_FILE_A));
PLIO* in2 = new PLIO("DataIn2", adf::plio_32_bits, QUOTE(INPUT_FILE_B));
PLIO* out1 = new PLIO("DataOut1", adf::plio_32_bits, QUOTE(OUTPUT_FILE));

simulation::platform<2, 1> platform(in1, in2, out1);
connect<> net0A(platform.src[0], matMult.inA[0]);
connect<> net0B(platform.src[1], matMult.inB[0]);
#else
const unsigned numInputs = P_CASC_LEN * 2;
// simulation::platform<numInputs,1> platform(INPUT_FILES_A , "data/inputB_0.txt", "data/inputB_1.txt",
// QUOTE(OUTPUT_FILE));

template <unsigned int numIn>
std::array<std::string, numIn> generateInputFilenames() {
    std::string inputFileAStr = QUOTE(INPUT_FILE_A);
    std::string inputFileBStr = QUOTE(INPUT_FILE_B);

    std::array<std::string, (numIn / 2)> inputFilesA;
    std::array<std::string, (numIn / 2)> inputFilesB;

    for (int i = 0; i < (numIn / 2); i++) {
        inputFilesA[i] = std::string(inputFileAStr);
        inputFilesA[i].insert(inputFilesA[i].length() - 4, ("_" + std::to_string(i)));

        inputFilesB[i] = std::string(inputFileBStr);
        inputFilesB[i].insert(inputFilesB[i].length() - 4, ("_" + std::to_string(i)));

        printf("Files to be read: \n%s, %s \n", inputFilesA[i].c_str(), inputFilesB[i].c_str());
    }
    std::array<std::string, numIn> allFiles;

    std::copy(inputFilesB.begin(), inputFilesB.end(),
              std::copy(inputFilesA.begin(), inputFilesA.end(), allFiles.begin()));
    return allFiles;
}

std::array<std::string, numInputs> allFiles = generateInputFilenames<numInputs>();
template <unsigned int numIn, typename Array, std::size_t... I>
auto createPlatformImpl(const Array& a, std::index_sequence<I...>) {
    return simulation::platform<numIn, 1>(a[I].c_str()..., QUOTE(OUTPUT_FILE));
}

template <unsigned int numIn, typename T, std::size_t N, typename Indices = std::make_index_sequence<N> >
auto createPlatform(const std::array<T, N>& a) {
    return createPlatformImpl<numIn>(a, Indices{});
}

auto platform = createPlatform<numInputs>(allFiles);

std::array<connect<>*, numInputs> net = [] {
    std::array<connect<>*, numInputs> ret;
    for (unsigned i = 0; i < P_CASC_LEN; ++i) {
        ret[i] = new connect<>(platform.src[i], matMult.inA[i]);
        ret[i + P_CASC_LEN] = new connect<>(platform.src[i + P_CASC_LEN], matMult.inB[i]);
    }
    return ret;
}();
#endif

#else
simulation::platform<2, 1> platform(QUOTE(INPUT_FILE_A), QUOTE(INPUT_FILE_B), QUOTE(OUTPUT_FILE));

connect<> net0A(platform.src[0], matMult.inA);
connect<> net0B(platform.src[1], matMult.inB);
#endif

connect<> outNet(matMult.out, platform.sink[0]);

#if defined(__AIESIM__) || defined(__X86SIM__)
int main(void) {
    printf("\n");
    // printf("%s %s\n", INPUT_FILES_A );
    printf("========================\n");
    printf("UUT: ");
    printf(QUOTE(UUT_GRAPH));
    printf("\n");
    printf("========================\n");
    printf("Input samples   = %d * %d \n", P_INPUT_SAMPLES_A, P_INPUT_SAMPLES_B);
    printf("Output samples  = %d \n", P_OUTPUT_SAMPLES);
    printf("DIM_A        = %d \n", P_DIM_A);
    printf("DIM_AB       = %d \n", P_DIM_AB);
    printf("DIM_B        = %d \n", P_DIM_B);
    printf("Shift           = %d \n", P_SHIFT);
    printf("ROUND_MODE      = %d \n", P_ROUND_MODE);
    printf("Data type A       = ");
    printf(QUOTE(T_DATA_A));
    printf("\n");
    printf("Data type B       = ");
    printf(QUOTE(T_DATA_B));
    printf("\n");

    matMult.init();
    matMult.run(NITER);
    matMult.end();

    return 0;
}
#endif
