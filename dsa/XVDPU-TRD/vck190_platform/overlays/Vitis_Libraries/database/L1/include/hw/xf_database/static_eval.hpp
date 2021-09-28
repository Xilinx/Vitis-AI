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

/**
 * @file static_eval.hpp
 * @brief static evaluation of user-privided function on each row.
 *
 * This file is part of Vitis Database Library.
 */

#ifndef XF_DATABASE_STATIC_EVAL_H
#define XF_DATABASE_STATIC_EVAL_H

#ifndef __cplusplus
#error "Vitis Database Library only works with C++."
#endif

#include "hls_math.h"
#include "hls_stream.h"

// for wide output
#include <ap_int.h>

namespace xf {
namespace database {
/**
 * @brief One stream input static evaluation
 *
 * static_eval function calculates the experssion result that user defined.
 * This result will be passed to aggregate module as the input.
 * When calling this API, the ``T`` ``T_O`` are the input/output data types
 * for each parameter of user code. E.g.
 *
 * \rst
 * ::
 *
 *     // decl
 *     long user_func(int a);
 *     // use
*      database::static_eval<int, long, user_func>(
 *       in1_strm, e_in_strm, out_strm, e_out_strm);
 *
 * \endrst
 *
 * In the above call, ``int`` is the data type of input of ``user_func``,
 * and ``long`` is the return type of ``user_func``.
 *
 * @tparam T the input stream type, inferred from argument
 * @tparam T_O the output stream type, inferred from argument
 * @tparam opf the user-defined expression function
 *
 * @param in_strm input data stream
 * @param e_in_strm end flag stream for input data
 * @param out_strm output data stream
 * @param e_out_strm end flag stream for output data
 */
template <typename T, typename T_O, T_O (*opf)(T)>
void staticEval(hls::stream<T>& in_strm,
                hls::stream<bool>& e_in_strm,
                hls::stream<T_O>& out_strm,
                hls::stream<bool>& e_out_strm) {
    bool e = e_in_strm.read();
    while (!e) {
#pragma HLS pipeline
        e = e_in_strm.read();
        T din = in_strm.read();
        T_O res = opf(din);
        out_strm.write(res);
        e_out_strm.write(0);
    }
    e_out_strm.write(1);
}

/**
 * @brief Two stream input static evaluation
 *
 * static_eval function calculate the experssion result that user defined.
 * This result will be passed to aggregate module as the input.
 * When calling this API, the ``T1`` ``T2`` ``T_O`` are the input/output data
* types
 * for each parameter of user code. E.g.
 *
 * \rst
 * ::
 *
 *     // decl
 *     long user_func(int a, int b);
 *     // use
*      database::static_eval<int, int, long, user_func>(
 *       in1_strm, in2_strm, e_in_strm, out_strm, e_out_strm);
 *
 * \endrst
 *
 * In the above call, two ``int`` are the data type of input of
* ``user_func``,
 * and ``long`` is the return type of ``user_func``.
 *
 * @tparam T1 the input stream type, inferred from argument
 * @tparam T2 the input stream type, inferred from argument
 * @tparam T_O the output stream type, inferred from argument
 * @tparam opf the user-defined expression function
 *
 * @param in1_strm input data stream
 * @param in2_strm input data stream
 * @param e_in_strm end flag stream for input data
 * @param out_strm output data stream
 * @param e_out_strm end flag stream for output data
 */
template <typename T1, typename T2, typename T_O, T_O (*opf)(T1, T2)>
void staticEval(hls::stream<T1>& in1_strm,
                hls::stream<T2>& in2_strm,
                hls::stream<bool>& e_in_strm,
                hls::stream<T_O>& out_strm,
                hls::stream<bool>& e_out_strm) {
    bool e = e_in_strm.read();
    while (!e) {
#pragma HLS pipeline
        e = e_in_strm.read();
        T1 in1 = in1_strm.read();
        T2 in2 = in2_strm.read();
        T_O res = opf(in1, in2);
        out_strm.write(res);
        e_out_strm.write(0);
    }
    e_out_strm.write(1);
}

/**
 * @brief Three stream input static evaluation
 *
 * static_eval function calculate the experssion result that user defined.
 * This result will be passed to aggregate module as the input.
 * When calling this API, the ``T1`` ``T2`` ``T3`` ``T_O`` are the input/output
* data types
 * for each parameter of user code. E.g.
 *
 * \rst
 * ::
 *
 *     // decl
 *     long user_func(int a, int b, int c);
 *     // use
*      database::static_eval<int, int, int, long, user_func>(
 *       in1_strm, in2_strm, in3_strm, e_in_strm,
 *       out_strm, e_out_strm);
 *
 * \endrst
 *
 * In the above call, three ``int`` are the data type of input of
* ``user_func``,
 * and ``long`` is the return type of ``user_func``.
 *
 * @tparam T1 the input stream type, inferred from argument
 * @tparam T2 the input stream type, inferred from argument
 * @tparam T3 the input stream type, inferred from argument
 * @tparam T_O the output stream type, inferred from argument
 * @tparam opf the user-defined expression function
 *
 * @param in1_strm input data stream
 * @param in2_strm input data stream
 * @param in3_strm input data stream
 * @param e_in_strm end flag stream for input data
 * @param out_strm output data stream
 * @param e_out_strm end flag stream for output data
 */
template <typename T1, typename T2, typename T3, typename T_O, T_O (*opf)(T1, T2, T3)>
void staticEval(hls::stream<T1>& in1_strm,
                hls::stream<T2>& in2_strm,
                hls::stream<T3>& in3_strm,
                hls::stream<bool>& e_in_strm,
                hls::stream<T_O>& out_strm,
                hls::stream<bool>& e_out_strm) {
    bool e = e_in_strm.read();
    while (!e) {
#pragma HLS pipeline
        e = e_in_strm.read();
        T1 in1 = in1_strm.read();
        T2 in2 = in2_strm.read();
        T3 in3 = in3_strm.read();
        T_O res = opf(in1, in2, in3);
        out_strm.write(res);
        e_out_strm.write(0);
    }
    e_out_strm.write(1);
}

/**
 * @brief Four stream input static evaluation
 *
 * static_eval function calculate the experssion result that user defined.
 * This result will be passed to aggregate module as the input.
 * When calling this API, the ``T1`` ``T2`` ``T3`` ``T_O`` are the input/output
* data types
 * for each parameter of user code. E.g.
 *
 * \rst
 * ::
 *
 *     // decl
 *     long user_func(int a, int b, int c, int d);
 *     // use
*      database::static_eval<int, int, int, int, long, user_func>(
 *       in1_strm, in2_strm, in3_strm, in3_strm, e_in_strm,
 *       out_strm, e_out_strm);
 *
 * \endrst
 *
 * In the above call, four ``int`` are the data type of input of
* ``user_func``,
 * and ``long`` is the return type of ``user_func``.
 *
 * @tparam T1 the input stream type, inferred from argument
 * @tparam T2 the input stream type, inferred from argument
 * @tparam T3 the input stream type, inferred from argument
 * @tparam T4 the input stream type, inferred from argument
 * @tparam T_O the output stream type, inferred from argument
 * @tparam opf the user-defined expression function
 *
 * @param in1_strm input data stream
 * @param in2_strm input data stream
 * @param in3_strm input data stream
 * @param in4_strm input data stream
 * @param e_in_strm end flag stream for input data
 * @param out_strm output data stream
 * @param e_out_strm end flag stream for output data
 */
template <typename T1, typename T2, typename T3, typename T4, typename T_O, T_O (*opf)(T1, T2, T3, T4)>
void staticEval(hls::stream<T1>& in1_strm,
                hls::stream<T2>& in2_strm,
                hls::stream<T3>& in3_strm,
                hls::stream<T4>& in4_strm,
                hls::stream<bool>& e_in_strm,
                hls::stream<T_O>& out_strm,
                hls::stream<bool>& e_out_strm) {
    bool e = e_in_strm.read();
    while (!e) {
#pragma HLS pipeline
        e = e_in_strm.read();
        T1 in1 = in1_strm.read();
        T2 in2 = in2_strm.read();
        T3 in3 = in3_strm.read();
        T4 in4 = in4_strm.read();
        T_O res = opf(in1, in2, in3, in4);
        out_strm.write(res);
        e_out_strm.write(0);
    }
    e_out_strm.write(1);
}

} // namespace database
} // namespace xf
#endif
