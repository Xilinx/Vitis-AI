/*
 * Copyright 2020 Xilinx, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WANCUNCUANTIES ONCU CONDITIONS OF ANY KIND, either express or
 * implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
/**
 * @file graph.hpp
 * @brief  This files contains graph definition.
 */

#ifndef _XILINXUDF_HPP_
#define _XILINXUDF_HPP_

#include "tgFunctions.hpp"
#include "loader.hpp"
#include "codevector.hpp"
#include <algorithm>

namespace UDIMPL {

/* Start Xilinx UDF additions */

inline double udf_bfs_fpga(int64_t sourceID,
                           ListAccum<int64_t>& offsetsList,
                           ListAccum<int64_t>& indicesList,
                           ListAccum<float>& weightsList,
                           ListAccum<int64_t>& predecent,
                           ListAccum<int64_t>& distance) {
    int numEdges = indicesList.size();
    int numVertices = offsetsList.size() - 1;
    uint32_t* predecentTmp = xf::graph::internal::aligned_alloc<uint32_t>(((numVertices + 15) / 16) * 16);
    uint32_t* distanceTmp = xf::graph::internal::aligned_alloc<uint32_t>(((numVertices + 15) / 16) * 16);
    memset(predecentTmp, -1, sizeof(uint32_t) * (((numVertices + 15) / 16) * 16));
    memset(distanceTmp, -1, sizeof(uint32_t) * (((numVertices + 15) / 16) * 16));
    xf::graph::Graph<uint32_t, uint32_t> g("CSR", numVertices, numEdges);

    int count = 0;
    while (count < numEdges) {
        if (count < offsetsList.size()) {
            int value0 = (int)(offsetsList.get(count));
            g.offsetsCSR[count] = value0;
        }
        int value = (int)(indicesList.get(count));
        float value1 = (float)(weightsList.get(count));
        g.indicesCSR[count] = value;
        g.weightsCSR[count] = 1;
        count++;
    }
    int res = bfs_fpga_wrapper(numVertices, numEdges, sourceID, g, predecentTmp, distanceTmp);

    for (int i = 0; i < numVertices; ++i) {
        if (predecentTmp[i] == (uint32_t)(-1)) {
            predecent += -1;
            distance += -1;
        } else {
            predecent += predecentTmp[i];
            distance += distanceTmp[i];
        }
    }
    g.freeBuffers();
    free(predecentTmp);
    free(distanceTmp);
    return res;
}

inline double udf_load_xgraph_fpga(ListAccum<int64_t>& offsetsList,
                                   ListAccum<int64_t>& indicesList,
                                   ListAccum<float>& weightsList) {
    int numEdges = indicesList.size();
    int numVertices = offsetsList.size() - 1;
    xf::graph::Graph<uint32_t, float> g("CSR", numVertices, numEdges);

    int count = 0;
    while (count < numEdges) {
        if (count < offsetsList.size()) {
            int value0 = (int)(offsetsList.get(count));
            g.offsetsCSR[count] = value0;
        }
        int value = (int)(indicesList.get(count));
        float value1 = (float)(weightsList.get(count));
        g.indicesCSR[count] = value;
        g.weightsCSR[count] = value1;
        count++;
    }
    int res = load_xgraph_fpga_wrapper(numVertices, numEdges, g);

    g.freeBuffers();
    return res;
}

inline double udf_shortest_ss_pos_wt_fpga(int64_t sourceID,
                                          int64_t numEdges,
                                          int64_t numVertices,
                                          ListAccum<int64_t>& predecent,
                                          ListAccum<float>& distance) {
    uint32_t length = ((numVertices + 1023) / 1024) * 1024;
    float** result;
    uint32_t** pred;
    result = new float*[1];
    pred = new uint32_t*[1];
    result[0] = xf::graph::internal::aligned_alloc<float>(length);
    pred[0] = xf::graph::internal::aligned_alloc<uint32_t>(length);
    memset(result[0], 0, length * sizeof(float));
    memset(pred[0], 0, length * sizeof(uint32_t));

    xf::graph::Graph<uint32_t, float> g("CSR", numVertices, numEdges);

    int res = shortest_ss_pos_wt_fpga_wrapper(numVertices, sourceID, 1, g, result, pred);

    for (int i = 0; i < numVertices; ++i) {
        predecent += pred[0][i];
        distance += result[0][i];
    }
    free(result[0]);
    free(pred[0]);
    delete[] result;
    delete[] pred;
    return res;
}

inline double udf_load_xgraph_pageRank_wt_fpga(ListAccum<int64_t>& offsetsList,
                                               ListAccum<int64_t>& indicesList,
                                               ListAccum<float>& weightsList) {
    int numEdges = indicesList.size();
    int numVertices = offsetsList.size() - 1;
    xf::graph::Graph<uint32_t, float> g("CSR", numVertices, numEdges);

    int count = 0;
    while (count < numEdges) {
        if (count < offsetsList.size()) {
            int value0 = (int)(offsetsList.get(count));
            g.offsetsCSR[count] = value0;
        }
        int value = (int)(indicesList.get(count));
        float value1 = (float)(weightsList.get(count));
        g.indicesCSR[count] = value;
        g.weightsCSR[count] = value1;
        count++;
    }
    int res = load_xgraph_pageRank_wt_fpga_wrapper(numVertices, numEdges, g);

    g.freeBuffers();
    return res;
}

inline double udf_pageRank_wt_fpga(
    int64_t numVertices, int64_t numEdges, float alpha, float tolerance, int64_t maxIter, ListAccum<float>& rank) {
    float* rankValue = new float[numVertices];

    xf::graph::Graph<uint32_t, float> g("CSR", numVertices, numEdges);

    int res = pageRank_wt_fpga_wrapper(alpha, tolerance, maxIter, g, rankValue);

    for (int i = 0; i < numVertices; ++i) {
        rank += rankValue[i];
    }
    delete[] rankValue;
    return res;
}

inline bool concat_uint64_to_str(string& ret_val, uint64_t val) {
    (ret_val += " ") += std::to_string(val);
    return true;
}

inline int64_t float_to_int_xilinx(float val) {
    return (int64_t)val;
}

inline int64_t udf_reinterpret_double_as_int64(double val) {
    int64_t double_to_int64 = *(reinterpret_cast<int64_t*>(&val));
    return double_to_int64;
}

inline double udf_reinterpret_int64_as_double(int64_t val) {
    double int64_to_double = *(reinterpret_cast<double*>(&val));
    return int64_to_double;
}

inline int64_t udf_lsb32bits(uint64_t val) {
    return val & 0x00000000FFFFFFFF;
}

inline int64_t udf_msb32bits(uint64_t val) {
    return (val >> 32) & 0x00000000FFFFFFFF;
}

inline VERTEX udf_getvertex(uint64_t vid) {
    return VERTEX(vid);
}

inline bool udf_setcode(int property, uint64_t startCode, uint64_t endCode, int64_t size) {
    return true;
}

inline bool udf_reset_timer(bool dummy) {
    return true;
}

inline double udf_elapsed_time(bool dummy) {
    return 1;
}

inline double udf_cos_theta(ListAccum<int64_t> vec_A, ListAccum<int64_t> vec_B) {
    double res;
    int size = vec_A.size();
    int64_t norm_A = vec_A.get(0);
    double norm_d_A = *(reinterpret_cast<double*>(&norm_A));
    int64_t norm_B = vec_B.get(0);
    double norm_d_B = *(reinterpret_cast<double*>(&norm_B));
    double prod = 0;
    int i = xai::startPropertyIndex;
    while (i < size) {
        prod = prod + vec_A.get(i) * vec_B.get(i);
        ++i;
    }
    res = prod / (norm_d_A * norm_d_B);
    std::cout << "val = " << res << std::endl;
    return res;
}

inline ListAccum<int64_t> udf_get_similarity_vec(int64_t property,
                                                 int64_t returnVecLength,
                                                 ListAccum<uint64_t>& property_vector) {
    ListAccum<uint64_t> result;
    int64_t size = property_vector.size();
    std::vector<uint64_t> codes;
    for (uint64_t val : property_vector) {
        codes.push_back(val);
    }
    std::vector<int> retcodes = xai::makeCosineVector(property, returnVecLength, codes);
    for (int value : retcodes) {
        result += value;
    }
    return result;
}

inline int udf_loadgraph_cosinesim_ss_fpga(int64_t numVertices,
                                           int64_t vecLength,
                                           ListAccum<ListAccum<int64_t> >& oldVectors) {
    xai::IDMap.clear();
    ListAccum<testResults> result;
    int32_t numEdges = vecLength - 3;

    const int splitNm = 3;    // kernel has 4 PUs, the input data should be splitted into 4 parts
    const int channelsPU = 4; // each PU has 4 HBM channels
    const int cuNm = 2;
    int deviceNeeded = 1;
    const int channelW = 16;
    int32_t nullVal = xai::NullVecValue;

    int32_t edgeAlign8 = ((numEdges + channelW - 1) / channelW) * channelW;
    int general = ((numVertices + deviceNeeded * cuNm * splitNm * channelsPU - 1) /
                   (deviceNeeded * cuNm * splitNm * channelsPU)) *
                  channelsPU;
    int rest = numVertices - general * (deviceNeeded * cuNm * splitNm - 1);
    if (rest < 0) {
        exit(1);
    }
    int32_t** numVerticesPU = new int32_t*[deviceNeeded * cuNm]; // vertex numbers in each PU
    int32_t** numEdgesPU = new int32_t*[deviceNeeded * cuNm];    // edge numbers in each PU

    int tmpID[deviceNeeded * cuNm * channelsPU * splitNm];
    for (int i = 0; i < deviceNeeded * cuNm; ++i) {
        numVerticesPU[i] = new int32_t[splitNm];
        numEdgesPU[i] = new int32_t[splitNm];
        for (int j = 0; j < splitNm; ++j) {
            numEdgesPU[i][j] = numEdges;
            for (int k = 0; k < channelsPU; ++k) {
                tmpID[i * splitNm * channelsPU + j * channelsPU + k] = 0;
            }
        }
    }
    //---------------- setup number of vertices in each PU ---------
    for (int i = 0; i < deviceNeeded * cuNm; ++i) {
        for (int j = 0; j < splitNm; ++j) {
            numVerticesPU[i][j] = general;
        }
    }
    numVerticesPU[deviceNeeded * cuNm - 1][splitNm - 1] = rest;

    xf::graph::Graph<int32_t, int32_t>** g = new xf::graph::Graph<int32_t, int32_t>*[deviceNeeded * cuNm];
    int fpgaNodeNm = 0;
    for (int i = 0; i < deviceNeeded * cuNm; ++i) {
        g[i] = new xf::graph::Graph<int32_t, int32_t>("Dense", 4 * splitNm, numEdges, numVerticesPU[i]);
        g[i][0].numEdgesPU = new int32_t[splitNm];
        g[i][0].numVerticesPU = new int32_t[splitNm];
        g[i][0].edgeNum = numEdges;
        g[i][0].nodeNum = numVertices;
        g[i][0].splitNum = splitNm;
        g[i][0].refID = fpgaNodeNm;
        for (int j = 0; j < splitNm; ++j) {
            fpgaNodeNm += numVerticesPU[i][j];
            int depth = ((numVerticesPU[i][j] + channelsPU - 1) / channelsPU) * edgeAlign8;
            g[i][0].numVerticesPU[j] = numVerticesPU[i][j];
            g[i][0].numEdgesPU[j] = depth;
            for (int l = 0; l < channelsPU; ++l) {
                for (int k = 0; k < depth; ++k) {
                    g[i][0].weightsDense[j * channelsPU + l][k] = nullVal;
                }
            }
        }
    }

    int offset = 0;
    for (int m = 0; m < deviceNeeded * cuNm; ++m) {
        for (int i = 0; i < splitNm; ++i) {
            int cnt[channelsPU] = {0};
            int subChNm = (numVerticesPU[m][i] + channelsPU - 1) / channelsPU;
            for (int j = 0; j < numVerticesPU[m][i]; ++j) {
                int64_t lsb32 = oldVectors.get(offset).get(1);
                int64_t msb32 = oldVectors.get(offset).get(2);
                uint64_t fullID = ((msb32 << 32) & 0xFFFFFFF00000000) | (lsb32 & 0x00000000FFFFFFFF);
                xai::IDMap.push_back(fullID);
                for (int k = 3; k < vecLength; ++k) {
                    g[m][0].weightsDense[i * channelsPU + j / subChNm][cnt[j / subChNm] * edgeAlign8 + k - 3] =
                        oldVectors.get(offset).get(k);
                }
                cnt[j / subChNm] += 1;
                offset++;
            }
        }
    }

    int ret = loadgraph_cosinesim_ss_dense_fpga_wrapper(deviceNeeded, cuNm, g);
    std::cout << "udf_loadgraph_cosinesim_ss_dense_fpga ret = " << ret << std::endl;
    return ret;
}

inline ListAccum<testResults> udf_cosinesim_ss_fpga(int64_t topK,
                                                    int64_t numVertices,
                                                    int64_t vecLength,
                                                    ListAccum<int64_t>& newVector) {
    ListAccum<testResults> result;
    int32_t numEdges = vecLength - 3;
    const int splitNm = 3;    // kernel has 4 PUs, the input data should be splitted into 4 parts
    const int channelsPU = 4; // each PU has 4 HBM channels
    const int cuNm = 2;
    int deviceNeeded = 1;
    const int channelW = 16;
    int32_t nullVal = xai::NullVecValue;

    int32_t edgeAlign8 = ((numEdges + channelW - 1) / channelW) * channelW;
    int general = ((numVertices + deviceNeeded * cuNm * splitNm * channelsPU - 1) /
                   (deviceNeeded * cuNm * splitNm * channelsPU)) *
                  channelsPU;
    int rest = numVertices - general * (deviceNeeded * cuNm * splitNm - 1);
    if (rest < 0) {
        exit(1);
    }
    int32_t** numVerticesPU = new int32_t*[deviceNeeded * cuNm]; // vertex numbers in each PU
    int32_t** numEdgesPU = new int32_t*[deviceNeeded * cuNm];    // edge numbers in each PU

    int tmpID[deviceNeeded * cuNm * channelsPU * splitNm];
    for (int i = 0; i < deviceNeeded * cuNm; ++i) {
        numVerticesPU[i] = new int32_t[splitNm];
        numEdgesPU[i] = new int32_t[splitNm];
        for (int j = 0; j < splitNm; ++j) {
            numEdgesPU[i][j] = numEdges;
            for (int k = 0; k < channelsPU; ++k) {
                tmpID[i * splitNm * channelsPU + j * channelsPU + k] = 0;
            }
        }
    }
    //---------------- setup number of vertices in each PU ---------
    for (int i = 0; i < deviceNeeded * cuNm; ++i) {
        for (int j = 0; j < splitNm; ++j) {
            numVerticesPU[i][j] = general;
        }
    }
    numVerticesPU[deviceNeeded * cuNm - 1][splitNm - 1] = rest;

    xf::graph::Graph<int32_t, int32_t>** g = new xf::graph::Graph<int32_t, int32_t>*[deviceNeeded * cuNm];
    int fpgaNodeNm = 0;
    for (int i = 0; i < deviceNeeded * cuNm; ++i) {
        g[i] = new xf::graph::Graph<int32_t, int32_t>("Dense", 4 * splitNm, numEdges, numVerticesPU[i]);
        g[i][0].numEdgesPU = new int32_t[splitNm];
        g[i][0].numVerticesPU = new int32_t[splitNm];
        g[i][0].edgeNum = numEdges;
        g[i][0].nodeNum = numVertices;
        g[i][0].splitNum = splitNm;
        g[i][0].refID = fpgaNodeNm;
        for (int j = 0; j < splitNm; ++j) {
            fpgaNodeNm += numVerticesPU[i][j];
            int depth = ((numVerticesPU[i][j] + channelsPU - 1) / channelsPU) * edgeAlign8;
            g[i][0].numVerticesPU[j] = numVerticesPU[i][j];
            g[i][0].numEdgesPU[j] = depth;
        }
    }
    //---------------- Generate Source Indice and Weight Array -------
    unsigned int sourceLen = edgeAlign8; // sourceIndice array length
    int32_t* sourceWeight =
        xf::graph::internal::aligned_alloc<int32_t>(sourceLen); // weights of source vertex's out members
    int32_t newVecLen = newVector.size() - 3;
    for (int i = 0; i < sourceLen; i++) {
        if (i < newVecLen) {
            sourceWeight[i] = newVector.get(i + 3);
        } else {
            sourceWeight[i] = nullVal;
        }
    }
    float* similarity = xf::graph::internal::aligned_alloc<float>(topK);
    int32_t* resultID = xf::graph::internal::aligned_alloc<int32_t>(topK);
    memset(resultID, 0, topK * sizeof(int32_t));
    memset(similarity, 0, topK * sizeof(float));

    //---------------- Run L3 API -----------------------------------
    int ret = cosinesim_ss_dense_fpga(deviceNeeded * cuNm, sourceLen, sourceWeight, topK, g, resultID, similarity);

    for (unsigned int k = 0; k < topK; k++) {
        result += testResults(VERTEX(xai::IDMap[resultID[k]]), similarity[k]);
    }

    return result;
}

inline double udf_close_fpga() {
    close_fpga();
    return 0;
}

/* End Xilinx Cosine Similarity Additions */
}

#endif /* EXPRFUNCTIONS_HPP_ */
