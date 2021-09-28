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
//================================== End Lic =================================================
// Printing Utility Functions !!
#ifndef SPU_H_
#define SPU_H_
#include <iostream>
#include <iomanip>
//#include "hls_ssr_fft_types.hpp"
//#include <DEBUG_CONSTANTS.hpp>
#include <complex>
//###################################### Utility Macros##################################START
#ifdef SHOW_DEC_MACRO_ENABLED
#define SHOW_DEC(a) std::cout << std::setw(10) << #a << std::dec << ":" << (a) << std::endl
#else
#define SHOW_DEC(a) ;
#endif
//###################################### Utility Macros####################################END

// Standard complex_wrapper support

template <int p_rows, int p_cols, typename T_dtype>
void SidebySidePrintComplexNumArrays(complex_wrapper<T_dtype> in[p_rows][p_cols],
                                     complex_wrapper<T_dtype> in2[p_rows][p_cols]) {
    std::cout << "\n\n################### Printing Arrays Side by Side for Comparison ##############" << std::endl;
    for (int i = 0; i < p_rows; i++)
        for (int j = 0; j < p_cols; j++) {
            // in[i][j] = i*(R)+j;
            std::cout << "[" << i << "]"
                      << "[" << j << "].real"
                      << "=" << std::setw(10) << in[i][j].real() << "  :  " << std::setw(10) << in2[i][j].real()
                      << " , ";
            std::cout << "[" << i << "]"
                      << "[" << j << "].imag"
                      << "=" << std::setw(10) << in[i][j].imag() << "  :  " << std::setw(10) << in2[i][j].imag()
                      << " , " << std::endl;
        }
    std::cout
        << "################### Element Wise Difference :: Printing Arrays Side by Side for Comparison ##############"
        << std::endl;

    for (int i = 0; i < p_rows; i++)
        for (int j = 0; j < p_cols; j++) {
            // in[i][j] = i*(R)+j;
            std::cout << "[" << i << "]"
                      << "[" << j << "].real.diff"
                      << "=" << std::setw(10) << in[i][j].real() - in2[i][j].real() << " , ";
            std::cout << "[" << i << "]"
                      << "[" << j << "].imag.diff"
                      << "=" << std::setw(10) << in[i][j].imag() - in2[i][j].imag() << " , " << std::endl;
        }
}

template <int p_rows, int p_cols, typename T_dtype1, typename T_dtype2>
void SidebySidePrintComplexNumArrays(complex_wrapper<T_dtype1> in[p_rows][p_cols],
                                     complex_wrapper<T_dtype2> in2[p_rows][p_cols]) {
    std::cout << "\n\n################### Printing Arrays Side by Side for Comparison ##############" << std::endl;
    for (int i = 0; i < p_rows; i++)
        for (int j = 0; j < p_cols; j++) {
            // in[i][j] = i*(R)+j;
            std::cout << "[" << i << "]"
                      << "[" << j << "].real"
                      << "=" << std::setw(10) << in[i][j].real() << "  :  " << std::setw(10) << in2[i][j].real()
                      << " , ";
            std::cout << "[" << i << "]"
                      << "[" << j << "].imag"
                      << "=" << std::setw(10) << in[i][j].imag() << "  :  " << std::setw(10) << in2[i][j].imag()
                      << " , " << std::endl;
        }
    std::cout
        << "################### Element Wise Difference :: Printing Arrays Side by Side for Comparison ##############"
        << std::endl;

    for (int i = 0; i < p_rows; i++)
        for (int j = 0; j < p_cols; j++) {
            // in[i][j] = i*(R)+j;
            std::cout << "[" << i << "]"
                      << "[" << j << "].real.diff"
                      << "=" << std::setw(10) << (double)in[i][j].real() - (double)in2[i][j].real() << " , ";
            std::cout << "[" << i << "]"
                      << "[" << j << "].imag.diff"
                      << "=" << std::setw(10) << (double)in[i][j].imag() - (double)in2[i][j].imag() << " , "
                      << std::endl;
        }
}

template <typename T_dtype, int p_rows, int p_cols>
void printComplexNumArray(complex_wrapper<T_dtype> in[p_rows][p_cols]) {
    std::cout << "###################" << std::endl;
    for (int j = 0; j < p_cols; j++)
        for (int i = 0; i < p_rows; i++) {
            // in[i][j] = i*(R)+j;
            std::cout << "[" << i << "]"
                      << "[" << j << "].real"
                      << "=" << in[i][j].real() << " , ";
            std::cout << "[" << i << "]"
                      << "[" << j << "].imag"
                      << "=" << in[i][j].imag() << std::endl;
            ;
        }
}

template <typename T_dtype, int p_rows>
void printComplexNumArray1D(complex_wrapper<T_dtype> in[p_rows]) {
    std::cout << "###################" << std::endl;
    for (int i = 0; i < p_rows; i++) {
        // in[i][j] = i*(R)+j;
        std::cout << "[" << i << "]"
                  << ".real"
                  << "=" << in[i].real() << " , ";
        std::cout << "[" << i << "]"
                  << ".imag"
                  << "=" << in[i].imag() << std::endl;
        ;
    }
}

template <int dim1, typename T_T>
void print1DArray(T_T data[dim1]) {
    std::cout << "\n################### Printing 1D Array ####################" << std::endl;
    for (int t = 0; t < dim1; t++) std::cout << std::setw(6) << t << "\t";
    std::cout << std::endl << "\n";

    for (int t = 0; t < dim1; t++) {
        std::cout << std::setw(6) << data[t] << "\t";
    }
    std::cout << std::endl;
}

template <int dim1, int dim2, typename T_T>
void print2DArray(T_T data[dim1][dim2]) {
    std::cout << "\n################### Printing 2D Array ####################" << std::endl;
    for (int t = 0; t < dim2; t++) std::cout << std::setw(6) << t << "\t";
    std::cout << std::endl;

    for (int t = 0; t < dim2 * 5; t++) std::cout << "-";
    std::cout << std::endl;
    for (int r = 0; r < dim1; r++) {
        for (int t = 0; t < dim2; t++) {
            std::cout << std::setw(6) << data[r][t] << "\t";
        }
        std::cout << std::endl;
    }
}

template <int dim1, int dim2, typename T_T>
void print2DArrayReal(T_T data[dim1][dim2]) {
    std::cout << "\n################### Printing 2D Array ####################" << std::endl;

    for (int t = 0; t < dim2; t++) std::cout << std::setw(6) << t << "\t";
    std::cout << std::endl << "\n";
    for (int r = 0; r < dim1; r++) {
        for (int t = 0; t < dim2; t++) {
            std::cout << std::setw(3) << data[r][t].real() << "\t";
        }
        std::cout << std::endl;
    }
}

#endif // !SPU_H_
