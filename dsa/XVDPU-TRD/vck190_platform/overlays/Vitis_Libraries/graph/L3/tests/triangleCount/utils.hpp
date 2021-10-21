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
 * @brief  This files contains graph definition and readin writeout graph.
 */

//#include <algorithm> //std::sort
#include <fstream>
#include <iostream>
//#include <iterator>
//#include <numeric> //std::iota
#include <sstream>
//#include <utility>
//#include <vector>

template <class ID_T>
void readInOffset(std::string filename, ID_T& numVertex, ID_T** offsetBuffer) {
    std::fstream fin(filename.c_str(), std::ios::in);
    if (!fin) {
        std::cout << "Error : file doesn't exist !" << std::endl;
        exit(1);
    }
    std::string line;
    int tmpNm;
    ID_T* offsetBufferTmp;
    while (std::getline(fin, line)) {
        if (line[0] != 37) {
            std::stringstream istr(line);
            istr >> tmpNm;
            numVertex = tmpNm;
            offsetBufferTmp = new ID_T[numVertex + 1];
            istr >> tmpNm;
            for (int i = 0; i < numVertex + 1; i++) {
                std::getline(fin, line);
                std::stringstream istr(line);
                ID_T row;
                istr >> row;
                offsetBufferTmp[i] = row;
            }
            break;
        };
    }
    fin.close();
    *offsetBuffer = offsetBufferTmp;
}

template <class ID_T, class VALUE_T>
void readInIndice(
    std::string filename, bool weighted, uint32_t& numEdges, ID_T** indiceBuffer, VALUE_T** weightBuffer) {
    std::fstream fin(filename.c_str(), std::ios::in);
    if (!fin) {
        std::cout << "Error : file doesn't exist !" << std::endl;
        exit(1);
    }
    std::string line;
    uint32_t tmpNm;
    ID_T* indiceBufferTmp;
    VALUE_T* weightBufferTmp;
    while (std::getline(fin, line)) {
        if (line[0] != 37) {
            std::stringstream istr(line);
            istr >> tmpNm;
            numEdges = tmpNm;
            indiceBufferTmp = new ID_T[numEdges];
            weightBufferTmp = new VALUE_T[numEdges];
            for (int i = 0; i < numEdges; i++) {
                std::getline(fin, line);
                std::stringstream istr(line);
                ID_T col;
                VALUE_T weight;
                istr >> col;
                if (!weighted) {
                    weight = 1.0;
                } else {
                    istr >> weight;
                }
                indiceBufferTmp[i] = col;
                weightBufferTmp[i] = weight;
            }
            break;
        };
    }
    fin.close();
    *indiceBuffer = indiceBufferTmp;
    *weightBuffer = weightBufferTmp;
}

template <class ID_T, class VALUE_T>
void readInCOO(std::string filename,
               bool weighted,
               uint32_t& numVertices,
               uint32_t& numEdges,
               ID_T** rowBuffer,
               ID_T** colBuffer,
               VALUE_T** weightBuffer) {
    std::fstream fin(filename.c_str(), std::ios::in);
    if (!fin) {
        std::cout << "Error : file doesn't exist !" << std::endl;
        exit(1);
    }
    std::string line;
    uint32_t tmpNm;
    ID_T* rowBufferTmp;
    ID_T* colBufferTmp;
    VALUE_T* weightBufferTmp;
    while (std::getline(fin, line)) {
        if (line[0] != 37) {
            std::stringstream istr(line);
            istr >> tmpNm;
            numVertices = tmpNm;
            istr >> tmpNm;
            numEdges = tmpNm;
            rowBufferTmp = new ID_T[numEdges];
            colBufferTmp = new ID_T[numEdges];
            weightBufferTmp = new VALUE_T[numEdges];
            for (int i = 0; i < numEdges; i++) {
                std::getline(fin, line);
                std::stringstream istr(line);
                ID_T row;
                ID_T col;
                VALUE_T weight;
                istr >> row;
                istr >> col;
                if (!weighted) {
                    weight = 1.0;
                } else {
                    istr >> weight;
                }
                rowBufferTmp[i] = row;
                colBufferTmp[i] = col;
                weightBufferTmp[i] = weight;
            }
            break;
        };
    }
    fin.close();
    *rowBuffer = rowBufferTmp;
    *colBuffer = colBufferTmp;
    *weightBuffer = weightBufferTmp;
}
