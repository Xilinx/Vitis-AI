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
 * WITHOUT WANCUNCUANTIES ONCU CONDITIONS OF ANY KIND, either express or
 * implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
/**
 * @file graph.hpp
 * @brief  This files contains graph definition and readin writeout graph.
 */

#ifndef _XF_GRAPH_HPP_
#define _XF_GRAPH_HPP_

#ifndef __SYNTHESIS__
#include <algorithm> //std::sort
#include <iostream>
#include <iterator>
#include <utility>
#include <vector>
#include <numeric> //std::iota
#include <fstream>
#include <sstream>

using namespace std;

template <typename T1, typename T2>
vector<int> sortInd(const vector<T1>& v1, const vector<T2>& v2) {
    vector<int> idx(v1.size());
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(), [&v1](int i1, int i2) { return v1[i1] < v1[i2]; });
    return idx;
}
template <typename T>
void StringSplit(std::vector<std::string>& dst, const std::string& src, const std::string& separator) {
    int nCount = 0;
    std::string temp;
    size_t pos = 0, offset = 0;

    while ((pos = src.find_first_of(separator, offset)) != std::string::npos) {
        temp = src.substr(offset, pos - offset);
        if (temp.length() > 0) {
            dst.push_back(temp);
        }
        offset = pos + 1;
    }

    temp = src.substr(offset, src.length() - offset);
    if (temp.length() > 0) {
        dst.push_back(temp);
    }
}

template <class ID_T, class VALUE_T>
class Node {
   public:
    ID_T index;
    VALUE_T value;
    Node(ID_T id) : index(id){};
    Node(ID_T id, VALUE_T v) : index(id), value(v){};
    void show(void) { std::cout << "Node.index: " << index << std::endl; }
};

template <class ID_T, class VALUE_T>
class Edge {
   public:
    ID_T start_node;
    ID_T end_node;
    // direction: 0, implies the egde is non-directed; 1 implies the egde is
    // directed.
    bool direction = 0;
    // weight
    VALUE_T weight = 0;
    Edge(ID_T n0, ID_T n1) : start_node(n0), end_node(n1){};
    Edge(ID_T n0, ID_T n1, VALUE_T w) : start_node(n0), end_node(n1), weight(w){};
    Edge(ID_T n0, ID_T n1, bool d, VALUE_T w) : start_node(n0), end_node(n1), direction(d), weight(w){};
    void show(void) {
        std::cout << "Edge.start_node: " << start_node << ", Edge.end_node: " << end_node << ", Edge.weight: " << weight
                  << std::endl;
    }
};

template <class ID_T, class VALUE_T>
class CooMatrix {
   public:
    vector<VALUE_T> value;
    vector<ID_T> row;
    vector<ID_T> column;
};

template <class ID_T, class VALUE_T>
class CsrMatrix {
   public:
    vector<VALUE_T> value;
    vector<ID_T> rowOffset;
    vector<ID_T> column;
};

template <class ID_T, class VALUE_T>
class CscMatrix {
   public:
    vector<VALUE_T> value;
    vector<ID_T> row;
    vector<ID_T> columnOffset;
};

template <class ForwardIt, class T>
void iota(ForwardIt first, ForwardIt last, T value) {
    while (first != last) {
        *first++ = value;
        ++value;
    }
}

template <class ID_T, class VALUE_T>
class Graph {
   public:
    int nodeNum;
    int edgeNum;

    vector<Node<ID_T, VALUE_T> > nodeListRcd;
    vector<Edge<ID_T, VALUE_T> > edgeListRcd;

    CooMatrix<ID_T, VALUE_T> cooMatrix;
    CsrMatrix<ID_T, VALUE_T> csrMatrix;
    CscMatrix<ID_T, VALUE_T> cscMatrix;

    int getNodeNum() {
        cout << "Node_Num = " << nodeListRcd.size() << endl;
        return nodeListRcd.size();
    }
    int getEdgeNum() {
        cout << "Edge_Num = " << cooMatrix.value.size() << endl;
        return edgeNum;
    }

    void addNode(Node<ID_T, VALUE_T> node) {
        nodeListRcd.push_back(node.index);
#ifdef __GRAPHDEBUG__
        node.show();
#endif
    }

    void addEdge(Edge<ID_T, VALUE_T> edge) {
        // COO
        cooMatrix.value.push_back(edge.weight);
        cooMatrix.row.push_back(edge.start_node);
        cooMatrix.column.push_back(edge.end_node);
#ifdef __GRAPHDEBUG__
        edge.show();
#endif
    }

    CooMatrix<ID_T, VALUE_T> getCooMatrix() {
#ifdef __GRAPHDEBUG__
        for (int j = 0; j < cooMatrix.value.size(); j++) {
            cout << "COO: value=" << cooMatrix.value[j] << ", row=" << cooMatrix.row[j]
                 << ", column=" << cooMatrix.column[j] << endl;
        }
#endif
        return cooMatrix;
    }

    template <typename T>
    class CompareIdByVectorValues {
        std::vector<T> _values;

       public:
        CompareIdByVectorValues(std::vector<T> values) : _values(values) {}

       public:
        bool operator()(const int& a, const int& b) const { return (_values)[a] < (_values)[b]; }
    };

    template <typename T>
    class CompareIdBy2VectorValues {
        std::vector<T> _values1;
        std::vector<T> _values2;

       public:
        CompareIdBy2VectorValues(std::vector<T> values1, std::vector<T> values2)
            : _values1(values1), _values2(values2) {}

       public:
        bool operator()(const int& a, const int& b) const {
            if ((_values1)[a] == (_values1)[b]) {
                return (_values2)[a] < (_values2)[b];
            } else {
                return (_values1)[a] < (_values1)[b];
            }
        }
    };

    template <typename T1, typename T2>
    vector<int> sortIndexes(const vector<T1>& v1, const vector<T2>& v2) {
        vector<int> idx(v1.size());
        for (size_t i = 0; i < std::distance(idx.begin(), idx.end()); i++) {
            idx.begin()[i] = v1[i];
        }
        // boost::sort::spreadsort::integer_sort(idx.begin(), idx.end());
        return idx;
    }

    CsrMatrix<ID_T, VALUE_T> getUndirectedMatrix() {
        vector<int> sortedIdx(cooMatrix.row.size());
        sortedIdx = sortIndexes<ID_T, ID_T>(cooMatrix.column, cooMatrix.row); // for undirected
        int offsetIndex = (cooMatrix.row[sortedIdx[0]] > cooMatrix.column[sortedIdx[0]])
                              ? cooMatrix.column[sortedIdx[0]]
                              : cooMatrix.row[sortedIdx[0]];
        ID_T preRowId = offsetIndex;
        ID_T curRowId = offsetIndex;

        csrMatrix.rowOffset.push_back(0);
        int ref = cooMatrix.column[sortedIdx[0] - offsetIndex] - offsetIndex;
        for (int m = 0; m < ref; ++m) {
            csrMatrix.rowOffset.push_back(0);
        }
        for (int j = 0; j < sortedIdx.size(); j++) {
            if (offsetIndex == 0) {
                csrMatrix.column.push_back(cooMatrix.row[j]);
            } else {
                csrMatrix.column.push_back(cooMatrix.row[j] - offsetIndex);
            }
            curRowId = sortedIdx[j];
            ID_T dist = curRowId - preRowId;
            for (int i = 0; i < dist; ++i) {
                csrMatrix.rowOffset.push_back(j);
                if (i == 0) {
                    preRowId = curRowId;
                }
            }
        }
        ID_T distNode = nodeNum - curRowId;
        for (int i = 0; i < distNode; ++i) {
            csrMatrix.rowOffset.push_back(sortedIdx.size());
        }
        csrMatrix.rowOffset.push_back(sortedIdx.size());

#ifdef __GRAPHDEBUG__
        // print CsrMatrix output
        for (int j = 0; j < csrMatrix.rowOffset.size() - 1; j++) {
            for (int m = csrMatrix.rowOffset[j]; m < csrMatrix.rowOffset[j + 1]; m++) {
                cout << "csr_row=" << j << ", csr_column=" << csrMatrix.column[m]
                     << ", csr_value=" << csrMatrix.value[m] << endl;
            }
        }
#endif

        return csrMatrix;
    }

    CsrMatrix<ID_T, VALUE_T> getCsrMatrix() {
        vector<int> sortedIdx(cooMatrix.row.size());
        sortedIdx = sortIndexes<ID_T, ID_T>(cooMatrix.column, cooMatrix.row); // for undirected

        int offsetIndex = (cooMatrix.row[sortedIdx[0]] > cooMatrix.column[sortedIdx[0]])
                              ? cooMatrix.column[sortedIdx[0]]
                              : cooMatrix.row[sortedIdx[0]];
        ID_T preRowId = offsetIndex;
        ID_T curRowId = offsetIndex;

        csrMatrix.rowOffset.push_back(0);
        int ref = cooMatrix.row[sortedIdx[0] - offsetIndex] - offsetIndex;
        for (int m = 0; m < ref; ++m) {
            csrMatrix.rowOffset.push_back(0);
        }
        for (int j = 0; j < sortedIdx.size(); j++) {
            if (offsetIndex == 0) {
                csrMatrix.column.push_back(cooMatrix.column[sortedIdx[j]]);
            } else {
                csrMatrix.column.push_back(cooMatrix.column[sortedIdx[j] - offsetIndex] - offsetIndex);
            }
            curRowId = cooMatrix.row[sortedIdx[j]] - offsetIndex;
            ID_T dist = curRowId - preRowId;
            for (int i = 0; i < dist; ++i) {
                csrMatrix.rowOffset.push_back(j);
                if (i == 0) {
                    preRowId = curRowId;
                }
            }
        }
        ID_T distNode = nodeNum - curRowId;
        for (int i = 0; i < distNode; ++i) {
            csrMatrix.rowOffset.push_back(sortedIdx.size());
        }
        csrMatrix.rowOffset.push_back(sortedIdx.size());

#ifdef __GRAPHDEBUG__
        // print CsrMatrix output
        for (int j = 0; j < csrMatrix.rowOffset.size() - 1; j++) {
            for (int m = csrMatrix.rowOffset[j]; m < csrMatrix.rowOffset[j + 1]; m++) {
                cout << "csr_row=" << j << ", csr_column=" << csrMatrix.column[m]
                     << ", csr_value=" << csrMatrix.value[m] << endl;
            }
        }
#endif

        return csrMatrix;
    }

    CscMatrix<ID_T, VALUE_T> getCscMatrix() {
        vector<int> sortedIdx(cooMatrix.row.size());
        sortedIdx = sortIndexes<ID_T, ID_T>(cooMatrix.column, cooMatrix.row);
        int offsetIndex = (cooMatrix.row[sortedIdx[0]] > cooMatrix.column[sortedIdx[0]])
                              ? cooMatrix.column[sortedIdx[0]]
                              : cooMatrix.row[sortedIdx[0]];
        ID_T preRowId = offsetIndex;
        ID_T curRowId = offsetIndex;

        cscMatrix.columnOffset.push_back(0);
        int ref = cooMatrix.column[sortedIdx[0] - offsetIndex] - offsetIndex;
        for (int m = 0; m < ref; ++m) {
            cscMatrix.columnOffset.push_back(0);
        }
        for (int j = 0; j < sortedIdx.size(); j++) {
            if (offsetIndex == 0) {
                cscMatrix.row.push_back(cooMatrix.row[sortedIdx[j]]);
            } else {
                cscMatrix.row.push_back(cooMatrix.row[j] - offsetIndex);
            }
            curRowId = sortedIdx[j];
            ID_T dist = curRowId - preRowId;
            for (int i = 0; i < dist; ++i) {
                cscMatrix.columnOffset.push_back(j);
                if (i == 0) {
                    preRowId = curRowId;
                }
            }
        }
        ID_T distNode = nodeNum - curRowId;
        for (int i = 0; i < distNode; ++i) {
            cscMatrix.columnOffset.push_back(sortedIdx.size());
        }
        cscMatrix.columnOffset.push_back(sortedIdx.size());

#ifdef __GRAPHDEBUG__
        // print CsrMatrix output
        for (int j = 0; j < cscMatrix.columnOffset.size() - 1; j++) {
            for (int m = cscMatrix.columnOffset[j]; m < cscMatrix.columnOffset[j + 1]; m++) {
                cout << "csc_column=" << j << ", csc_row=" << cscMatrix.row[m] << endl;
            }
        }
#endif
        return cscMatrix;
    }
}; // end class Graph;

template <class ID_T, class VALUE_T>
void readInWeightedDirectedGraphOffset(std::string filename, CscMatrix<ID_T, VALUE_T>& cscMat, int nnz, int& nrows) {
    std::fstream fin(filename.c_str(), std::ios::in);
    if (!fin) {
        std::cout << "Error : file doesn't exist !" << std::endl;
        exit(1);
    }
    std::string line;
    int tmpNm;
    while (std::getline(fin, line)) {
        if (line[0] != 37) {
            std::stringstream istr(line);
            istr >> tmpNm;
            nrows = tmpNm;
            for (int i = 0; i < nrows; i++) {
                std::getline(fin, line);
                std::stringstream istr(line);
                ID_T tmp;
                istr >> tmp;
                cscMat.columnOffset.push_back(tmp);
            }
            cscMat.columnOffset.push_back(nnz);
            break;
        };
    }
    fin.close();
}

template <class ID_T, class VALUE_T>
void readInWeightedDirectedGraphCV(std::string filename, CscMatrix<ID_T, VALUE_T>& cscMat, int& nnz) {
    std::fstream fin(filename.c_str(), std::ios::in);
    if (!fin) {
        std::cout << "Error : file doesn't exist !" << std::endl;
        exit(1);
    }
    std::string line;
    int tmpNm;
    while (std::getline(fin, line)) {
        if (line[0] != 37) {
            std::stringstream istr(line);
            istr >> tmpNm;
            nnz = tmpNm;
            int j = 0;
            for (int i = 0; i < nnz; i++) {
                std::getline(fin, line);
                std::stringstream istr(line);
                j = 0;
                VALUE_T tmp;
                ID_T col;
                while (istr >> tmp) {
                    switch (j) {
                        case 0:
                            col = tmp;
                            break;
                    }
                    ++j;
                    if (j > 2) break;
                }
                cscMat.row.push_back(col);
                cscMat.value.push_back(tmp);
            }
            if (j == 1)
                std::cout << "Warning : Weight doesn't exist ! Use default weight=1.0 need undefine WEIGHTED_GRAPH in "
                             "host code"
                          << std::endl;
            break;
        };
    }
    fin.close();
}

template <class ID_T, class VALUE_T>
void writeOutWeightedDirectedGraphRow(std::string filename, CscMatrix<ID_T, VALUE_T>& cscMat, int nrows) {
    std::fstream fin(filename.c_str(), std::ios::out);
    if (!fin) {
        std::cout << "Error : file doesn't exist !" << std::endl;
        exit(1);
    }
    fin << nrows << "\n";
    for (int i = 0; i < nrows + 1; ++i) {
        fin << cscMat.columnOffset[i] << "\n";
    }
    fin.close();
}

template <class ID_T, class VALUE_T>
void writeOutWeightedDirectedGraphCV(std::string filename, CscMatrix<ID_T, VALUE_T>& cscMat, int nnz) {
    std::fstream fin(filename.c_str(), std::ios::out);
    if (!fin) {
        std::cout << "Error : file doesn't exist !" << std::endl;
        exit(1);
    }
    fin << nnz << "\n";
    for (int i = 0; i < nnz; ++i) {
        fin << cscMat.row[i] << "\n";
    }
    fin.close();
}

template <class ID_T, class VALUE_T>
void writeOutWeightedDirectedGraphCV(std::string filename, CsrMatrix<ID_T, VALUE_T>& csrMat, int nnz) {
    std::fstream fin(filename.c_str(), std::ios::out);
    if (!fin) {
        std::cout << "Error : file doesn't exist !" << std::endl;
        exit(1);
    }
    fin << nnz << "\n";
    for (int i = 0; i < nnz; ++i) {
        fin << csrMat.column[i] << " " << csrMat.value[i] << "\n";
    }
    fin.close();
}

template <class ID_T, class VALUE_T>
void readInWeightedDirectedGraph(std::string filename, Graph<ID_T, VALUE_T>& g) {
    std::fstream fin(filename.c_str(), std::ios::in);
    if (!fin) {
        std::cout << "Error : file doesn't exist !" << std::endl;
        exit(1);
    }
    std::string line;
    int tmpNm;
    while (std::getline(fin, line)) {
        if (line[0] != 37) {
            std::stringstream istr(line);
            istr >> tmpNm;
            g.nodeNum = tmpNm;
            istr >> tmpNm;
            istr >> tmpNm;
            g.edgeNum = tmpNm;
            int nnz = tmpNm;
            for (int i = 0; i < nnz; i++) {
                std::getline(fin, line);
                std::stringstream istr(line);
                int j = 0;
                VALUE_T tmp;
                ID_T row, col;
                while (istr >> tmp) {
                    switch (j) {
                        case 0:
                            row = tmp;
                            break;
                        case 1:
                            col = tmp;
                            break;
                        case 2:
                            break;
                    }
                    ++j;
                    if (j > 3) break;
                }
                Edge<ID_T, VALUE_T> edge(row, col, tmp);
                g.addEdge(edge);
            }
            break;
        };
    }
    fin.close();
}

template <class ID_T, class VALUE_T>
void readInRef(std::string filename1, VALUE_T* out, int nrows) {
    std::fstream fin1(filename1.c_str(), std::ios::in);
    if (!fin1) {
        std::cout << "Error : file doesn't exist !" << std::endl;
        exit(1);
    }
    std::string line;
    int tmpNm;
    int cnt = 0;
    while (std::getline(fin1, line)) {
        std::vector<std::string> tmpVec;
        StringSplit<int>(tmpVec, line, " ");
        VALUE_T tmp;
        tmp = std::atof(tmpVec[0].c_str());
        out[cnt] = tmp;
        cnt++;
    }
    fin1.close();
}

template <class ID_T, class VALUE_T>
void readInRef(std::string filename1,
               std::string filename2,
               std::string filename3,
               std::string filename4,
               std::vector<ID_T> row,
               std::vector<VALUE_T> value,
               VALUE_T* out,
               int nrows) {
    std::fstream fin1(filename1.c_str(), std::ios::in);
    std::fstream fin2(filename2.c_str(), std::ios::in);
    std::fstream fin3(filename3.c_str(), std::ios::in);
    std::fstream fin4(filename4.c_str(), std::ios::in);
    if ((!fin1) || (!fin2) || (!fin3) || (!fin4)) {
        std::cout << "Error : file doesn't exist !" << std::endl;
        exit(1);
    }
    std::string line;
    int tmpNm;
    while (std::getline(fin1, line)) {
        std::vector<std::string> tmpVec;
        StringSplit<int>(tmpVec, line, " ");
        VALUE_T tmp;
        ID_T row1;
        row1 = std::stoi(tmpVec[0]);
        tmp = std::atof(tmpVec[1].c_str());
        row.push_back(row1 - 1);
        value.push_back(tmp);
    }
    fin1.close();
    while (std::getline(fin2, line)) {
        std::vector<std::string> tmpVec;
        StringSplit<int>(tmpVec, line, " ");
        VALUE_T tmp;
        ID_T row1;
        row1 = std::stoi(tmpVec[0]);
        tmp = std::atof(tmpVec[1].c_str());
        row.push_back(row1 - 1);
        value.push_back(tmp);
    }
    fin2.close();
    while (std::getline(fin3, line)) {
        std::vector<std::string> tmpVec;
        StringSplit<int>(tmpVec, line, " ");
        VALUE_T tmp;
        ID_T row1;
        row1 = std::stoi(tmpVec[0]);
        tmp = std::atof(tmpVec[1].c_str());
        row.push_back(row1 - 1);
        value.push_back(tmp);
    }
    fin3.close();
    while (std::getline(fin4, line)) {
        std::vector<std::string> tmpVec;
        StringSplit<int>(tmpVec, line, " ");
        VALUE_T tmp;
        ID_T row1;
        row1 = std::stoi(tmpVec[0]);
        tmp = std::atof(tmpVec[1].c_str());
        row.push_back(row1 - 1);
        value.push_back(tmp);
    }
    fin4.close();
    std::vector<int> sortedIdx(nrows);
    sortedIdx = sortInd<ID_T, VALUE_T>(row, value);
    for (int i = 0; i < row.size(); ++i) {
        out[row[sortedIdx[i]]] = value[sortedIdx[i]];
    }
}

template <class ID_T, class VALUE_T>
void readInTigerRef(std::string filename1, std::vector<ID_T> row, std::vector<VALUE_T> value, VALUE_T* out, int nrows) {
    std::fstream fin1(filename1.c_str(), std::ios::in);
    if (!fin1) {
        std::cout << "Error : file doesn't exist !" << std::endl;
        exit(1);
    }
    std::string line;
    int tmpNm;
    std::getline(fin1, line); // throw the first row
    std::cout << "get first line !" << std::endl;
    while (std::getline(fin1, line)) {
        std::vector<std::string> tmpVec;
        StringSplit<int>(tmpVec, line, ",");
        VALUE_T tmp;
        ID_T row1;
        row1 = std::stoi(tmpVec[0]);
        tmp = std::atof(tmpVec[1].c_str());
        row.push_back(row1 - 1);
        value.push_back(tmp);
    }
    fin1.close();

    std::vector<int> sortedIdx(nrows);
    sortedIdx = sortInd<ID_T, VALUE_T>(row, value);
    std::cout << "row.size() = " << row.size() << std::endl;
    std::cout << "sortedIdx.size() = " << sortedIdx.size() << std::endl;

    for (int i = 0; i < row.size(); ++i) {
        out[row[sortedIdx[i]]] = value[sortedIdx[i]];
    }
}

#endif //#ifndef __SYNTHESIS__
#endif //#ifndef XF_GRAPH_HPP_
