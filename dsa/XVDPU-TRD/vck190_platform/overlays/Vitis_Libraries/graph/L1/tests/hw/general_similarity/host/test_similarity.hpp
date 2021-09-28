/*
 * Copyright 2019 Xilinx, Inc.
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

#ifndef XF_GRAPH_TEST_SIMILARITY_HPP
#define XF_GRAPH_TEST_SIMILARITY_HPP

#include "utils.hpp"

void generateSourceParams(unsigned int numVerticesPU[PU_NUMBER],
                          unsigned int numEdges,
                          int dataType,
                          int sourceID,
                          ap_uint<32>* offset32[PU_NUMBER],
                          ap_uint<32>* indice32[PU_NUMBER],
                          float* weightSparse[PU_NUMBER],
                          unsigned int& sourceNUM,
                          ap_uint<32>** sourceIndice,
                          ap_uint<32>** sourceWeight) {
    int row0, col0, row1, col1;

    bool flag0 = 0;
    int k;
    int sumV = 0;
    for (k = 0; k < PU_NUMBER; ++k) {
        sumV += numVerticesPU[k];
        if (sourceID < sumV) {
            flag0 = 1;
            break;
        }
    }
    if (flag0) {
        row0 = k;
        row1 = k;

        col0 = sourceID + 1 + numVerticesPU[k] - sumV;
        col1 = sourceID + numVerticesPU[k] - sumV;
    } else {
        row0 = PU_NUMBER - 1;
        row1 = PU_NUMBER - 1;

        col0 = numVerticesPU[PU_NUMBER - 1] + 1;
        col1 = numVerticesPU[PU_NUMBER - 1];
    }
    std::cout << "sourceID=" << sourceID << " offset0=" << offset32[row0][col0] << " offset1=" << offset32[row1][col1]
              << std::endl;
    std::cout << "row0=" << row0 << " col0=" << col0 << " row1=" << row1 << " col1=" << col1 << std::endl;
    sourceNUM = (unsigned int)(offset32[row0][col0] - offset32[row1][col1]);
    *sourceIndice = aligned_alloc<ap_uint<32> >(4096);
    *sourceWeight = aligned_alloc<ap_uint<32> >(4096);

    int ind = offset32[row1][col1] - offset32[row1][0];
    for (int i = 0; i < sourceNUM; i++) {
        sourceIndice[0][i] = (ap_uint<32>)indice32[row1][ind + i];
        sourceWeight[0][i] = floatToBits<float, uint32_t>(weightSparse[row1][ind + i]);

        std::cout << "sourceWeight[" << i << "]=" << sourceWeight[0][i]
                  << " weightSparse=" << weightSparse[row1][ind + i] << std::endl;
    }
}

void generateSourceParams(unsigned int numVerticesPU[PU_NUMBER],
                          unsigned int numEdges,
                          int dataType,
                          int sourceID,
                          float* weightDense[4 * PU_NUMBER],
                          unsigned int& sourceNUM,
                          ap_uint<32>** sourceWeight) {
    sourceNUM = (unsigned int)numEdges;
    *sourceWeight = aligned_alloc<ap_uint<32> >(4096);

    unsigned int id, row;
    unsigned int offset[4 * PU_NUMBER + 1];
    offset[0] = 0;
    for (int i = 0; i < 4 * PU_NUMBER; i++) {
        offset[i + 1] = numVerticesPU[i / 4] + offset[i];
        if ((sourceID >= offset[i]) && (sourceID < offset[i + 1])) {
            id = i;
            row = sourceID - offset[i];
        }
    }

    std::cout << "id =" << id << " row=" << row << std::endl;
    for (int i = 0; i < sourceNUM; i++) {
        sourceWeight[0][i] = floatToBits<float, uint32_t>(weightDense[id][row * numEdges + i]);

        std::cout << "sourceWeight[" << i << "]=" << sourceWeight[0][i] << std::endl;
    }
}

int computeSimilarity(std::string xclbinPath,
                      std::string goldenFile,
                      unsigned int numVertices,
                      unsigned int numEdges,
                      int similarityType,
                      int dataType,
                      int sourceID,
                      int sortK,
                      int repInt,
                      unsigned int numVerticesPU[PU_NUMBER],
                      unsigned int numEdgesPU[PU_NUMBER],
                      ap_uint<32>* offset32[PU_NUMBER],
                      ap_uint<32>* indice32[PU_NUMBER],
                      float* weightSparse[PU_NUMBER],
                      unsigned int sourceNUM,
                      ap_uint<32>* sourceIndice,
                      ap_uint<32>* sourceWeight) {
    struct timeval start_time; // End to end time clock start
    gettimeofday(&start_time, 0);

    // output && config////////////////////////////////////////////////////////////////
    std::vector<ap_uint<32>*> config(repInt);
    std::vector<ap_uint<32>*> result_id(repInt);
    std::vector<float*> similarity(repInt);
    unsigned int startID[PU_NUMBER];
    unsigned int tmp = 0;
    for (int i = 0; i < PU_NUMBER - 1; i++) { // calculate multi PU start address
        startID[i] = tmp;
        tmp += numVerticesPU[i];
    }
    startID[PU_NUMBER - 1] = tmp;
    for (int i = 0; i < PU_NUMBER; i++) {
        std::cout << "startID[" << i << "]=" << startID[i] << std::endl;
    }
    for (int i = 0; i < repInt; i++) {
        similarity[i] = aligned_alloc<float>(128);
        result_id[i] = aligned_alloc<ap_uint<32> >(128);
        int base_id = 3;
        config[i] = aligned_alloc<ap_uint<32> >(64);
        config[i][0] = sortK;
        config[i][1] = sourceNUM;
        config[i][2] = similarityType;
        config[i][3] = dataType;
        config[i][4] = 0;

        for (int j = 0; j < PU_NUMBER; j++) {
            config[i][5 + j] = startID[j];
            config[i][5 + PU_NUMBER + j] = numVerticesPU[j];
            config[i][5 + 2 * PU_NUMBER + j] = numEdgesPU[j];
        }
    }

    generalSimilarityKernel(config[0], sourceIndice, sourceWeight, (ap_uint<32 * CHANNEL_NUMBER>*)offset32[0],
                            (ap_uint<32 * CHANNEL_NUMBER>*)indice32[0], (ap_uint<32 * CHANNEL_NUMBER>*)weightSparse[0],
                            (ap_uint<32 * CHANNEL_NUMBER>*)weightSparse[0], (ap_uint<32 * CHANNEL_NUMBER>*)offset32[1],
                            (ap_uint<32 * CHANNEL_NUMBER>*)indice32[1], (ap_uint<32 * CHANNEL_NUMBER>*)weightSparse[1],
                            (ap_uint<32 * CHANNEL_NUMBER>*)weightSparse[1], (ap_uint<32 * CHANNEL_NUMBER>*)offset32[2],
                            (ap_uint<32 * CHANNEL_NUMBER>*)indice32[2], (ap_uint<32 * CHANNEL_NUMBER>*)weightSparse[2],
                            (ap_uint<32 * CHANNEL_NUMBER>*)weightSparse[2], (ap_uint<32 * CHANNEL_NUMBER>*)offset32[3],
                            (ap_uint<32 * CHANNEL_NUMBER>*)indice32[3], (ap_uint<32 * CHANNEL_NUMBER>*)weightSparse[3],
                            (ap_uint<32 * CHANNEL_NUMBER>*)weightSparse[3], result_id[0], similarity[0]);

    // need to write a compare function in order to compare golden values with results and put it here
    ////////////////////////////////////////////////////////////////////////////////////////////////////////
    int err = checkData<PU_NUMBER>(goldenFile, result_id[0], similarity[0]);

    return err;
}

int computeSimilarity(std::string xclbinPath,
                      std::string goldenFile,
                      unsigned int numVertices,
                      unsigned int numEdges,
                      int similarityType,
                      int dataType,
                      int sourceID,
                      int sortK,
                      int repInt,
                      unsigned int numVerticesPU[PU_NUMBER],
                      unsigned int numEdgesPU[PU_NUMBER],
                      float* weightDense[4 * PU_NUMBER],
                      unsigned int sourceNUM,
                      ap_uint<32>* sourceWeight) {
    struct timeval start_time; // End to end time clock start
    gettimeofday(&start_time, 0);

    // output && config////////////////////////////////////////////////////////////////
    std::vector<ap_uint<32>*> config(repInt);
    std::vector<ap_uint<32>*> result_id(repInt);
    std::vector<float*> similarity(repInt);
    unsigned int startID[PU_NUMBER];
    unsigned int tmp = 0;
    for (int i = 0; i < PU_NUMBER - 1; i++) { // calculate multi PU start address
        startID[i] = tmp;
        tmp += 4 * numVerticesPU[i];
    }
    startID[PU_NUMBER - 1] = tmp;
    for (int i = 0; i < repInt; i++) {
        similarity[i] = aligned_alloc<float>(128);
        result_id[i] = aligned_alloc<ap_uint<32> >(128);
        int base_id = 3;
        config[i] = aligned_alloc<ap_uint<32> >(64);
        config[i][0] = sortK;
        config[i][1] = sourceNUM;
        config[i][2] = similarityType;
        config[i][3] = dataType;
        config[i][4] = 1;

        for (int j = 0; j < PU_NUMBER; j++) {
            config[i][5 + j] = startID[j];
            config[i][5 + PU_NUMBER + j] = numVerticesPU[j];
            config[i][5 + 2 * PU_NUMBER + j] = numEdgesPU[j];
        }
    }

    generalSimilarityKernel(
        config[0], sourceWeight, sourceWeight, (ap_uint<32 * CHANNEL_NUMBER>*)weightDense[0],
        (ap_uint<32 * CHANNEL_NUMBER>*)weightDense[1], (ap_uint<32 * CHANNEL_NUMBER>*)weightDense[2],
        (ap_uint<32 * CHANNEL_NUMBER>*)weightDense[3], (ap_uint<32 * CHANNEL_NUMBER>*)weightDense[4],
        (ap_uint<32 * CHANNEL_NUMBER>*)weightDense[5], (ap_uint<32 * CHANNEL_NUMBER>*)weightDense[6],
        (ap_uint<32 * CHANNEL_NUMBER>*)weightDense[7], (ap_uint<32 * CHANNEL_NUMBER>*)weightDense[8],
        (ap_uint<32 * CHANNEL_NUMBER>*)weightDense[9], (ap_uint<32 * CHANNEL_NUMBER>*)weightDense[10],
        (ap_uint<32 * CHANNEL_NUMBER>*)weightDense[11], (ap_uint<32 * CHANNEL_NUMBER>*)weightDense[12],
        (ap_uint<32 * CHANNEL_NUMBER>*)weightDense[13], (ap_uint<32 * CHANNEL_NUMBER>*)weightDense[14],
        (ap_uint<32 * CHANNEL_NUMBER>*)weightDense[15], result_id[0], similarity[0]);

    // need to write a compare function in order to compare golden values with results and put it here
    ////////////////////////////////////////////////////////////////////////////////////////////////////////
    int err = checkData<PU_NUMBER>(goldenFile, result_id[0], similarity[0]);

    return err;
}

#endif //#ifndef VT_GRAPH_SIMILARITY_H
