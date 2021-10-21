#ifndef _XF_SOLVER_MATRIX_UTILITY_
#define _XF_SOLVER_MATRIX_UTILITY_

#include <iostream>
#include <iomanip>
#include <fstream>
#include <string.h>
#include <sstream>
#include <bitset>
//#include "hls_math.h"
#include <math.h>
//#include "util.h"

//////// write out matrix[MM][NN] to the file with the name of file_name ///////
template <typename dataType>
void writeOut(int row, int col, dataType** outMat, std::string& filename) {
    std::ofstream outfile;
    outfile.open(filename, std::ios::binary);
    if (!outfile.is_open()) {
        std::cout << "Output file could not open: " << filename << std::endl;
        return;
    } else {
        std::cout << "Output file: " << filename << std::endl;
    }
    outfile.write(reinterpret_cast<char*>(&row), sizeof(int));
    outfile.write(reinterpret_cast<char*>(&col), sizeof(int));
    for (int i = 0; i < row; ++i) {
        for (int j = 0; j < col; ++j) {
            dataType in = outMat[i][j];
            outfile.write(reinterpret_cast<char*>(&in), sizeof(dataType));
        }
    }
    outfile.close();
}

//////// read_in matrix[MM][NN] from the file with the name of file_name ///////
template <typename dataType>
void readIn(int& row, int& col, dataType** inMat, std::string& filename) {
    std::ifstream infile;
    infile.open(filename, std::ios::binary);
    if (!infile.is_open()) {
        std::cout << "Error opening file" << std::endl;
        exit(1);
    }
    infile.read(reinterpret_cast<char*>(&row), sizeof(int));
    infile.read(reinterpret_cast<char*>(&col), sizeof(int));
    for (int i = 0; i < row; ++i) {
        for (int j = 0; j < col; ++j) {
            dataType in;
            infile.read(reinterpret_cast<char*>(&in), sizeof(dataType));
            inMat[i][j] = in;
        }
    }
    infile.close();
}

////// generate general random matrix ////////////////////
template <typename dataType>
void matGen(int m, int n, unsigned int seed, dataType** matrix) {
    srand(seed);
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            matrix[i][j] = (dataType)((rand() % (10000)) / (0.7 + n + seed / 7));
        }
    }
}

////// generate general random matrix ////////////////////
template <typename dataType>
void matGen(int m, int n, unsigned int seed, dataType* matrix) {
    srand(seed);
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            matrix[i * n + j] = (dataType)((rand() % (10000)) / (0.7 + n + seed / 7));
        }
    }
}

////// generate symmetric random matrix ////////////////////
template <typename dataType>
void symMatGen(int n, unsigned int seed, dataType** matrix) {
    srand(seed);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j <= i; j++) {
            matrix[i][j] = (dataType)(rand() % (10000)) * 1.0 / n;
            if (i != j) {
                matrix[j][i] = matrix[i][j];
            }
        }
    }
}
template <typename dataType>
void symMatGen(int n, unsigned int seed, dataType* matrix) {
    srand(seed);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j <= i; j++) {
            matrix[i * n + j] = (dataType)(rand() % (10000)) * 1.0 / n;
            if (i != j) {
                matrix[j * n + i] = matrix[i * n + j];
            }
        }
    }
}

/////////  generate triangular lower random matrix used for SPD matrix Gen /////////////
template <typename dataType>
void triLowerMatGenSPD(int n, unsigned int seed, dataType** matrix) {
    srand(seed);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j <= i; j++) {
            matrix[i][j] = (dataType)(rand() % (10000)) * 1.0 / n;
            if (i != j) {
                matrix[j][i] = 0;
            }
            if (i == j) {
                while (matrix[i][i] == 0) {
                    matrix[i][j] = (dataType)(rand() % (10000)) * 1.0 / n;
                }
                matrix[j][i] *= n * 1.0;
            }
        }
    }
}

/////////  generate triangular lower random matrix /////////////
template <typename dataType>
void triLowerMatGen(int n, unsigned int seed, dataType** matrix) {
    srand(seed);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j <= i; j++) {
            matrix[i][j] = (dataType)(rand() % (10000)) * 1.0 / n;
            if (i != j) {
                matrix[j][i] = 0;
            }
            if (i == j) {
                matrix[j][i] *= n * 1.0;
            }
        }
    }
}

////////   matrix transpose  //////////////////////////////
template <typename dataType>
void transposeMat(int n, dataType** matIn, dataType** matOut) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            matOut[j][i] = matIn[i][j];
        }
    }
}
template <typename dataType>
void transposeMat(int n, dataType* matIn, dataType* matOut) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            matOut[j * n + i] = matIn[i * n + j];
        }
    }
}

//////////  sum(array[NN])  ////////////////////////////////////
template <typename dataType>
dataType sumArray(int maxSize, dataType* array) {
    dataType tmpSum = 0;
    for (int i = 0; i < maxSize; ++i) {
        tmpSum += array[i];
    }
    return tmpSum;
}

//////// matrix multiplier of matrixLeft[MM][NN] * matrixRight[NN][P] //////
template <typename dataType>
void MulMat(int maxRow1, int maxCol1, int maxCol2, dataType** matLeft, dataType** matRight, dataType** matOut) {
    for (int i = 0; i < maxRow1; i++) {
        for (int j = 0; j < maxCol2; j++) {
            dataType buffer[maxCol1];
            for (int k = 0; k < maxCol1; k++) {
                buffer[k] = matLeft[i][k] * matRight[k][j];
            }
            matOut[i][j] = sumArray<dataType>(maxCol1, buffer);
        }
    }
}
template <typename dataType>
void MulMat(int maxRow1, int maxCol1, int maxCol2, dataType* matLeft, dataType* matRight, dataType* matOut) {
    for (int i = 0; i < maxRow1; i++) {
        for (int j = 0; j < maxCol2; j++) {
            dataType buffer[maxCol1];
            for (int k = 0; k < maxCol1; k++) {
                buffer[k] = matLeft[i * maxCol1 + k] * matRight[k * maxCol2 + j];
            }
            matOut[i * maxCol2 + j] = sumArray<dataType>(maxCol1, buffer);
        }
    }
}

//////// matrix multiplier of matrixLeft[maxRow1][maxCol1] * matrixMiddle[maxCol1][maxCol2] *
/// matrixRight[maxCol2][maxCol3] //////
template <typename dataType>
void MulMat(int maxRow1,
            int maxCol1,
            int maxCol2,
            int maxCol3,
            dataType* matLeft,
            dataType* matMiddle,
            dataType* matRight,
            dataType* matOut) {
    dataType* matTmp;
    matTmp = new dataType[maxRow1 * maxCol2];
    MulMat<dataType>(maxRow1, maxCol1, maxCol2, matLeft, matMiddle, matTmp);
    MulMat<dataType>(maxRow1, maxCol2, maxCol3, matTmp, matRight, matOut);
    delete[] matTmp;
}

//////// random diagonal matrix generator //////
template <typename dataType>
void diagonalMatGen(int n, unsigned int seed, dataType** matrix) {
    srand(seed);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (i == j) {
                matrix[i][i] = (dataType)(rand() % (10000)) * 1.0;
            } else {
                matrix[i][j] = 0;
            }
        }
    }
}

//////  print matrixIn[m][n] with header desc ///////
template <typename dataType>
void print_matrix(char* desc, int m, int n, dataType** matrixIn) {
    int i, j;
    printf("\n %s\n", desc);
    for (i = 0; i < m; i++) {
        for (j = 0; j < n; j++) printf("\t%6.30lf", matrixIn[i][j]);
        printf("\n");
    }
}

/////  print matrix[m*n] with header desc
template <typename dataType>
void print_matrix(char* desc, int m, int n, dataType* a, int lda) {
    int i, j;
    printf("\n %s\n", desc);
    for (i = 0; i < m; i++) {
        for (j = 0; j < n; j++) printf(" %6.30lf", a[j + i * lda]);
        printf("\n");
    }
}

/////  Functions used in QR
template <class T>
bool compareMatrices(T* m1, T* m2, int rows, int cols, int LD) {
    bool hasDiff = false;
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            T v1 = m1[i * LD + j];
            T v2 = m2[i * LD + j];
            if (fabs(v1 - v2) > 0.0001) {
                // std::cout << "row " << i << ", column " << j << " m1 "
                //          << v1 << " m2 " << v2 << std::endl;
                hasDiff = true;
            }
        }
    }

    return !hasDiff;
}

template <class T>
void getIdentityMatrix(T* matrix, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            if (i == j) {
                matrix[i * cols + j] = 1;
            } else {
                matrix[i * cols + j] = 0;
            }
        }
    }
}

template <class T>
void matrixMult(T* m1, int row1, int col1, T* m2, int row2, int col2, T* result) {
    if (col1 != row2) {
        return;
    }

    for (int i = 0; i < row1; ++i) {
        for (int j = 0; j < col2; ++j) {
            T sum = 0;
            for (int k = 0; k < col1; ++k) {
                sum += m1[i * col1 + k] * m2[k * col2 + j];
            }

            result[i * col2 + j] = sum;
        }
    }
}

template <class T>
void matrixSquareMultInline(T* m1, T* m2, int rows) {
    T* temp = new T[rows * rows];

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < rows; ++j) {
            T sum = 0;
            for (int k = 0; k < rows; ++k) {
                sum += m1[i * rows + k] * m2[k * rows + j];
            }

            temp[i * rows + j] = sum;
        }
    }

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < rows; ++j) {
            m1[i * rows + j] = temp[i * rows + j];
        }
    }

    delete[] temp;
}

template <class T>
void matrixSub(T* m1, T* m2, int row, int col, T* result) {
    for (int i = 0; i < row; ++i) {
        for (int j = 0; j < col; ++j) {
            int idx = i * col + j;
            result[idx] = m1[idx] - m2[idx];
        }
    }
}

template <class T>
void matrixFactorInline(T* matrix, T factor, int row, int col) {
    for (int i = 0; i < row; ++i) {
        for (int j = 0; j < col; ++j) {
            int idx = i * col + j;
            matrix[idx] *= factor;
        }
    }
}

template <class T>
bool matrixSupp(T* m1, T row, T col, T* m2, T targetRow, T targetCol) {
    if (targetRow < row || targetCol < col) {
        return false;
    }

    for (int i = 0; i < targetRow; ++i) {
        for (int j = 0; j < targetCol; ++j) {
            int idx = i * targetCol + j;
            if (i < targetRow - row || j < targetCol - col) {
                m2[idx] = 0;
            } else {
                int r = i - (targetRow - row);
                int c = j - (targetCol - col);
                int idx_ori = r * col + c;
                m2[idx] = m1[idx_ori];
            }
        }
    }

    return true;
}

template <class T>
void constructQ(T* result, T* tau, int rows, int cols, T* Q) {
    int num = std::min(rows, cols);
    getIdentityMatrix<T>(Q, rows, rows);

    for (int j = 0; j < num; ++j) {
        int length = rows - j;
        T* v = new T[rows - j];
        v[0] = 1;
        for (int i = 1; i < length; ++i) {
            int rowIdx = i + j;
            v[i] = result[rowIdx * cols + j];
        }

        T beta = tau[j];

        // get Q_k
        // Q_k = I - beta*v*v'
        // I: (rows - j) * (rows - j)
        // v*v'
        T* vv = new T[(rows - j) * (rows - j)];
        matrixMult<T>(v, rows - j, 1, v, 1, rows - j, vv);

        matrixFactorInline<T>(vv, beta, rows - j, rows - j);

        T* vvAll = new T[rows * rows];
        matrixSupp<T>(vv, rows - j, rows - j, vvAll, rows, rows);

        T* I = new T[rows * rows];
        getIdentityMatrix<T>(I, rows, rows);

        T* Qk = new T[rows * rows];
        matrixSub<T>(I, vvAll, rows, rows, Qk);

        matrixSquareMultInline<T>(Q, Qk, rows);
    }
}

template <class T>
void convertToRInline(T* data, int row, int col) {
    for (int i = 0; i < row; ++i) {
        for (int j = 0; j < col; ++j) {
            if (i > j) {
                data[i * col + j] = 0.0;
            }
        }
    }
}

#endif
