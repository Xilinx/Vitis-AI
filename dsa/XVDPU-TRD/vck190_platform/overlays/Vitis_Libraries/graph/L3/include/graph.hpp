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

#pragma once

#ifndef _XF_GRAPH_L3_GRAPH_HPP_
#define _XF_GRAPH_L3_GRAPH_HPP_

#include <iostream>
#include <vector>
#include <algorithm>
#include <memory.h>

namespace xf {
namespace graph {
namespace internal {
template <typename MType>
union f_cast2;

template <>
union f_cast2<double> {
    double f;
    int64_t i;
};

template <>
union f_cast2<float> {
    float f;
    int32_t i;
};
template <typename T, typename T2>
class sortHelper {
   public:
    sortHelper(T a, T2 b) : first(a), second(b) {}
    T first;
    T2 second;
    bool operator<(const sortHelper& m) const { return first < m.first; }
};

template <typename T, typename T2>
void indexedSort(uint32_t length, std::vector<T>& p, std::vector<T2>& q) {
    std::vector<sortHelper<T, T2> > vec;
    for (int i = 0; i < p.size(); ++i) {
        sortHelper<T, T2> cell(p[i], q[i]);
        vec.push_back(cell);
    }
    std::sort(vec.begin(), vec.end());
    for (int i = 0; i < p.size(); ++i) {
        p[i] = vec[i].first;
        q[i] = vec[i].second;
    }
    vec.erase(vec.begin(), vec.end());
}

template <typename T>
T* aligned_alloc(std::size_t num) {
    void* ptr = nullptr;
#if _WIN32
    ptr = (T*)malloc(num * sizeof(T));
    if (num == 0) {
#else
    if (posix_memalign(&ptr, 4096, num * sizeof(T))) {
#endif
        throw std::bad_alloc();
    }
    return reinterpret_cast<T*>(ptr);
}
}

template <class ID_T, class VALUE_T>
class Graph {
   private:
    enum graphType { COO = 1, CSR = 2, CSC = 3, Dense = 4 } type;
    bool allocatedCOO = 0;
    bool allocatedCSR = 0;
    bool allocatedCSC = 0;
    bool allocatedDense = 0;

   public:
    ID_T nodeNum;
    ID_T edgeNum;

    ID_T axiNm;
    ID_T refID = 0;
    VALUE_T** weightsDense;

    ID_T* rowsCOO;
    ID_T* colsCOO;
    VALUE_T* weightsCOO;

    ID_T nrowCSR;
    ID_T* offsetsCSR;
    ID_T* indicesCSR;
    VALUE_T* weightsCSR;

    ID_T nrowCSC;
    ID_T* offsetsCSC;
    ID_T* indicesCSC;
    VALUE_T* weightsCSC;

    ID_T splitNum;
    ID_T** offsetsSplitted;
    ID_T** indicesSplitted;
    VALUE_T** weightsSplitted;
    ID_T* numEdgesPU;
    ID_T* numVerticesPU;
    bool splitted = 0;

    // ---------------- construction with numVertices and numEdges ----------------
    /**
     * @brief Constructor of class dense Graph
     *
     * @param numElementsPU elements number of each PU
     * @param numEdges edges number of Graph
     * @param axis axi numbers
    */

    Graph(std::string type, ID_T axis, ID_T numEdges, ID_T* numVerticesPU) {
        ID_T channelsW = 16;
        if (type == "Dense") {
            axiNm = axis;
            weightsDense = new VALUE_T*[axiNm];
            int edgeAlign8 = ((numEdges + channelsW - 1) / channelsW) * channelsW;
            for (int i = 0; i < axiNm / 4; ++i) {
                int depth = (numVerticesPU[i] + 3) / 4 * edgeAlign8;
                for (int j = 0; j < 4; ++j) {
                    weightsDense[i * 4 + j] = internal::aligned_alloc<VALUE_T>(depth);
                    memset(weightsDense[i * 4 + j], 0, depth * sizeof(VALUE_T));
                }
            }
            allocatedDense = 1;
        }
    }

    Graph(std::string type, ID_T axis, ID_T* numElementsPU) {
        if (type == "Dense") {
            axiNm = axis;
            weightsDense = new VALUE_T*[axiNm];
            for (int i = 0; i < axiNm; ++i) {
                weightsDense[i] = internal::aligned_alloc<VALUE_T>(numElementsPU[i / 4]);
            }
            allocatedDense = 1;
        }
    }

    void allocSplittedOffsets(ID_T numSplit, ID_T* numNodePU) {
        splitNum = numSplit;
        splitted = 1;
        offsetsSplitted = new ID_T*[splitNum];
        indicesSplitted = new ID_T*[splitNum];
        weightsSplitted = new VALUE_T*[splitNum];
        numEdgesPU = new ID_T[splitNum];
        numVerticesPU = new ID_T[splitNum];
        // ------------- split offsets --------------------------------------------
        unsigned int sumVertex = 0;
        for (int i = 0; i < numSplit; i++) {
            sumVertex += numNodePU[i];
            numVerticesPU[i] = numNodePU[i];
            offsetsSplitted[i] = internal::aligned_alloc<ID_T>(numVerticesPU[i] + 1);
        }
        if (sumVertex != nodeNum) { // vertex numbers between file input and numVerticesPU should match
            std::cout << "Error: sum of PU vertex numbers doesn't match file input vertex number, sumVertex = "
                      << sumVertex << "\t nodeNum = " << nodeNum << std::endl;
            exit(1);
        }
    }

    void allocSplittedIndicesWeights() {
        // ------------- calculate numEdgesPU from offset arrays ------------------
        for (int i = 0; i < splitNum; i++) {
            if (i < splitNum - 1) {
                numEdgesPU[i] = offsetsSplitted[i + 1][0] - offsetsSplitted[i][0];
            } else {
                numEdgesPU[splitNum - 1] = edgeNum - offsetsSplitted[splitNum - 1][0];
            }
        }
        // ------------- split indices & weight -----------------------------------
        for (int i = 0; i < splitNum; i++) {
            indicesSplitted[i] = internal::aligned_alloc<ID_T>(numEdgesPU[i]);
            weightsSplitted[i] = internal::aligned_alloc<VALUE_T>(numEdgesPU[i]);
        }
    }

    void splitCSR(ID_T numSplit, ID_T* numNodePU) {
        splitNum = numSplit;
        splitted = 1;
        offsetsSplitted = new ID_T*[splitNum];
        indicesSplitted = new ID_T*[splitNum];
        weightsSplitted = new VALUE_T*[splitNum];
        numEdgesPU = new ID_T[splitNum];
        numVerticesPU = new ID_T[splitNum];
        ID_T counter = 0;
        // ------------- split offsets --------------------------------------------
        unsigned int sumVertex = 0;
        for (int i = 0; i < numSplit; i++) {
            sumVertex += numNodePU[i];
            numVerticesPU[i] = numNodePU[i];
            offsetsSplitted[i] = internal::aligned_alloc<ID_T>(numVerticesPU[i] + 1);
            for (int j = 0; j < numVerticesPU[i]; ++j) {
                offsetsSplitted[i][j] = offsetsCSR[counter];
                counter++;
            }
            offsetsSplitted[i][numVerticesPU[i]] = offsetsCSR[counter];
        }
        if (sumVertex != nodeNum) { // vertex numbers between file input and numVerticesPU should match
            std::cout << "Error: sum of PU vertex numbers doesn't match file input vertex number, sumVertex = "
                      << sumVertex << "\t nodeNum = " << nodeNum << std::endl;
            exit(1);
        }
        // ------------- calculate numEdgesPU from offset arrays ------------------
        for (int i = 0; i < splitNum; i++) {
            if (i < splitNum - 1) {
                numEdgesPU[i] = offsetsSplitted[i + 1][0] - offsetsSplitted[i][0];
            } else {
                numEdgesPU[splitNum - 1] = edgeNum - offsetsSplitted[splitNum - 1][0];
            }
        }
        counter = 0;
        // ------------- split indices & weight -----------------------------------
        for (int i = 0; i < splitNum; i++) {
            indicesSplitted[i] = internal::aligned_alloc<ID_T>(numEdgesPU[i] + 1);
            weightsSplitted[i] = internal::aligned_alloc<VALUE_T>(numEdgesPU[i] + 1);
            for (int j = 0; j < numEdgesPU[i]; ++j) {
                indicesSplitted[i][j] = indicesCSR[counter];
                weightsSplitted[i][j] = weightsCSR[counter];
                counter++;
            }
        }
    }

    // ---------------- construction with numVertices and numEdges ----------------
    /**
     * @brief Constructor of class Graph
     *
     * @param numVertices nodes number of Graph
     * @param numEdges edges number of Graph
    */

    Graph(std::string type, ID_T numVertices, ID_T numEdges) {
        nodeNum = numVertices;
        edgeNum = numEdges;
        nrowCSR = numVertices + 1;
        nrowCSC = numVertices + 1;
        if (type == "CSR") {
            offsetsCSR = internal::aligned_alloc<ID_T>(nrowCSR);
            indicesCSR = internal::aligned_alloc<ID_T>(numEdges);
            weightsCSR = internal::aligned_alloc<VALUE_T>(numEdges);
            allocatedCSR = 1;
        } else if (type == "CSC") {
            offsetsCSC = internal::aligned_alloc<ID_T>(nrowCSC);
            indicesCSC = internal::aligned_alloc<ID_T>(numEdges);
            weightsCSC = internal::aligned_alloc<VALUE_T>(numEdges);
            allocatedCSC = 1;
        } else if (type == "COO") {
            rowsCOO = internal::aligned_alloc<ID_T>(numEdges);
            colsCOO = internal::aligned_alloc<ID_T>(numEdges);
            weightsCOO = internal::aligned_alloc<VALUE_T>(numEdges);
            allocatedCOO = 1;
        }
    }

    // ---------------- construction with existant buffers ------------------------
    /**
     * @brief Constructor of class Graph with existant buffers
     *
     * @param numVertices nodes number of Graph
     * @param numEdges edges number of Graph
     * @param buffer1 COO: rows buffer, CSR&CSC: offsets buffer
     * @param buffer2 COO: cols buffer, CSR&CSC: indices buffer
     * @param buffer3 COO&CSR&CSC: weights buffer
    */

    Graph(std::string type, ID_T numVertices, ID_T numEdges, ID_T* buffer1, ID_T* buffer2, VALUE_T* buffer3) {
        nodeNum = numVertices;
        edgeNum = numEdges;
        nrowCSR = numVertices + 1;
        nrowCSC = numVertices + 1;
        if (type == "CSR") {
            offsetsCSR = internal::aligned_alloc<ID_T>(nrowCSR);
            indicesCSR = internal::aligned_alloc<ID_T>(numEdges);
            weightsCSR = internal::aligned_alloc<VALUE_T>(numEdges);
            for (int i = 0; i < edgeNum; ++i) {
                if (i < nrowCSR) {
                    offsetsCSR[i] = buffer1[i];
                }
                indicesCSR[i] = buffer2[i];
                weightsCSR[i] = buffer3[i];
            }
            allocatedCSR = 1;
        } else if (type == "CSC") {
            offsetsCSC = internal::aligned_alloc<ID_T>(nrowCSC);
            indicesCSC = internal::aligned_alloc<ID_T>(numEdges);
            weightsCSC = internal::aligned_alloc<VALUE_T>(numEdges);
            for (int i = 0; i < edgeNum; ++i) {
                if (i < nrowCSC) {
                    offsetsCSC[i] = buffer1[i];
                }
                indicesCSC[i] = buffer2[i];
                weightsCSC[i] = buffer3[i];
            }
            allocatedCSC = 1;
        } else if (type == "COO") {
            rowsCOO = internal::aligned_alloc<ID_T>(numEdges);
            colsCOO = internal::aligned_alloc<ID_T>(numEdges);
            weightsCOO = internal::aligned_alloc<VALUE_T>(numEdges);
            for (int i = 0; i < edgeNum; ++i) {
                rowsCOO[i] = buffer1[i];
                colsCOO[i] = buffer2[i];
                weightsCOO[i] = buffer3[i];
            }
            allocatedCOO = 1;
        }
    }

    void allocBuffer(std::string type, ID_T maxNodes, ID_T maxEdges) {
        if (type == "CSR") {
            if (allocatedCSR) {
                free(offsetsCSR);
                free(indicesCSR);
                free(weightsCSR);
            }
            offsetsCSR = internal::aligned_alloc<ID_T>(maxNodes);
            indicesCSR = internal::aligned_alloc<ID_T>(maxEdges);
            weightsCSR = internal::aligned_alloc<VALUE_T>(maxEdges);
            allocatedCSR = 1;
        } else if (type == "CSC") {
            if (allocatedCSC) {
                free(offsetsCSC);
                free(indicesCSC);
                free(weightsCSC);
            }
            offsetsCSC = internal::aligned_alloc<ID_T>(maxNodes);
            indicesCSC = internal::aligned_alloc<ID_T>(maxEdges);
            weightsCSC = internal::aligned_alloc<VALUE_T>(maxEdges);
            allocatedCSC = 1;
        } else if (type == "COO") {
            if (allocatedCOO) {
                free(rowsCOO);
                free(colsCOO);
                free(weightsCOO);
            }
            rowsCOO = internal::aligned_alloc<ID_T>(maxEdges);
            colsCOO = internal::aligned_alloc<ID_T>(maxEdges);
            weightsCOO = internal::aligned_alloc<VALUE_T>(maxEdges);
            allocatedCOO = 1;
        }
    }

    void COO2CSR() {
        if (allocatedCOO) {
            std::vector<ID_T>* inMember = new std::vector<ID_T>[ nodeNum ];
            std::vector<ID_T>* outMember = new std::vector<ID_T>[ nodeNum ];
            std::vector<VALUE_T>* inWeights = new std::vector<VALUE_T>[ nodeNum ];
            std::vector<VALUE_T>* outWeights = new std::vector<VALUE_T>[ nodeNum ];
            for (int i = 0; i < edgeNum; ++i) {
                int src = rowsCOO[i];
                int des = colsCOO[i];
                VALUE_T weight = weightsCOO[i];
                outMember[src].push_back(des);
                inMember[des].push_back(src);
                outWeights[src].push_back(weight);
                inWeights[des].push_back(weight);
            }

            if (!allocatedCSR) {
                offsetsCSR = internal::aligned_alloc<ID_T>(nrowCSR);
                indicesCSR = internal::aligned_alloc<ID_T>(edgeNum);
                weightsCSR = internal::aligned_alloc<VALUE_T>(edgeNum);
                allocatedCSR = 1;
            }

            ID_T offset = 0;
            for (int i = 0; i < nodeNum; ++i) {
                offsetsCSR[i] = offset;
                ID_T num = outMember[i].size();
                internal::indexedSort<ID_T, VALUE_T>(num, outMember[i], outWeights[i]);
                for (int j = 0; j < num; ++j) {
                    indicesCSR[offset + j] = outMember[i][j];
                    weightsCSR[offset + j] = outWeights[i][j];
                }
                offset += num;
            }
            offsetsCSR[nodeNum] = edgeNum;

            delete[] inMember;
            delete[] outMember;
            delete[] inWeights;
            delete[] outWeights;
        } else {
            std::cout << "Error: No available COO graph found" << std::endl;
        }
    }

    void CSC2CSR() {
        if (allocatedCSC) {
            std::vector<ID_T>* inMember = new std::vector<ID_T>[ nodeNum ];
            std::vector<ID_T>* outMember = new std::vector<ID_T>[ nodeNum ];
            std::vector<VALUE_T>* inWeights = new std::vector<VALUE_T>[ nodeNum ];
            std::vector<VALUE_T>* outWeights = new std::vector<VALUE_T>[ nodeNum ];
            ID_T offset = 0;
            for (int i = 0; i < nodeNum; ++i) {
                offset = offsetsCSC[i];
                ID_T num = offsetsCSC[i + 1] - offsetsCSC[i];
                for (int j = 0; j < num; ++j) {
                    int des = i;
                    int src = indicesCSC[offset + j];
                    VALUE_T weight = weightsCSC[offset + j];
                    outMember[src].push_back(des);
                    inMember[des].push_back(src);
                    outWeights[src].push_back(weight);
                    inWeights[des].push_back(weight);
                }
            }

            if (!allocatedCSR) {
                offsetsCSR = internal::aligned_alloc<ID_T>(nrowCSR);
                indicesCSR = internal::aligned_alloc<ID_T>(edgeNum);
                weightsCSR = internal::aligned_alloc<VALUE_T>(edgeNum);
                allocatedCSR = 1;
            }

            offset = 0;
            for (int i = 0; i < nodeNum; ++i) {
                offsetsCSR[i] = offset;
                ID_T num = outMember[i].size();
                internal::indexedSort<ID_T, VALUE_T>(num, outMember[i], outWeights[i]);
                for (int j = 0; j < num; ++j) {
                    indicesCSR[offset + j] = outMember[i][j];
                    weightsCSR[offset + j] = outWeights[i][j];
                }
                offset += num;
            }
            offsetsCSR[nodeNum] = edgeNum;

            delete[] inMember;
            delete[] outMember;
            delete[] inWeights;
            delete[] outWeights;
        } else {
            std::cout << "Error: No available CSC graph found" << std::endl;
        }
    }

    void freeBuffers() {
        if (allocatedCSC) {
            free(offsetsCSC);
            free(indicesCSC);
            free(weightsCSC);
        }
        if (allocatedCSR) {
            free(offsetsCSR);
            free(indicesCSR);
            free(weightsCSR);
        }
        if (allocatedCOO) {
            free(rowsCOO);
            free(colsCOO);
            free(weightsCOO);
        }
        if (allocatedDense) {
            for (int i = 0; i < axiNm; ++i) {
                free(weightsDense[i]);
            }
            delete[] weightsDense;
        }
        if (splitted) {
            for (int i = 0; i < splitNum; i++) { // free splitted buffer
                free(offsetsSplitted[i]);
                free(indicesSplitted[i]);
                free(weightsSplitted[i]);
            }
            delete[] offsetsSplitted;
            delete[] indicesSplitted;
            delete[] weightsSplitted;
            splitted = 0;
        }
    }

}; // end class Graph;

} // end of graph
} // end of xf

#endif //#ifndef XF_GRAPH_L3_GRAPH_HPP_
