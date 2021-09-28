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
 * @file dynamic_eval.hpp
 * @brief This file contains run-time-configurable expression evaluation
 * primitive.
 *
 * This file is part of Vitis Database Library.
 */

#ifndef XF_DATABASE_DYNAMIC_EVAL_H
#define XF_DATABASE_DYNAMIC_EVAL_H

#ifndef __cplusplus
#error "Vitis Database Library only works with C++."
#endif

#include <ap_int.h>
#include <hls_stream.h>

//#define _XFDB_DYN_EVAL_DEBUG true

namespace xf {
namespace database {
namespace details {

//------------------Dynamic ALU Hardware------------------

// implement mul && add
template <typename In_Type1, typename In_Type2, typename Out_Type>
Out_Type alu_math(In_Type1 strm_in1, In_Type2 strm_in2, ap_uint<3> OP) {
#pragma HLS INLINE

    Out_Type in1;
    Out_Type in2;
    Out_Type result_temp1;
    Out_Type result_temp2;
    Out_Type result;

    // control sign of strm_in1
    if (OP(1, 1)) {
        in1 = -strm_in1;
    } else {
        in1 = strm_in1;
    }

    // control sign of strm_in2
    if (OP(0, 0)) {
        in2 = -strm_in2;
    } else {
        in2 = strm_in2;
    }

    result_temp1 = in1 * in2;
    result_temp2 = in1 + in2;

    // control result
    if (OP(2, 2)) {
        result = result_temp1;
    } else {
        result = result_temp2;
    }

    return result;
}

// implement comparator
template <typename In_Type1, typename In_Type2>
bool alu_comparator(In_Type1 strm_in1, In_Type2 strm_in2, ap_uint<3> OP) {
#pragma HLS INLINE

    bool result;

    if (strm_in1 > strm_in2) {
        switch (OP) {
            case 0:
                result = 1;
                break; // strm_in1>strm_in2
            case 1:
                result = 1;
                break; // strm_in1>=strm_in2
            case 2:
                result = 0;
                break; // strm_in1==strm_in2
            case 3:
                result = 1;
                break; // strm_in1!=strm_in2
            case 4:
                result = 0;
                break; // strm_in1<=strm_in2
            case 5:
                result = 0;
                break; // strm_in1<strm_in2
        }
    } else if (strm_in1 == strm_in2) {
        switch (OP) {
            case 0:
                result = 0;
                break; // strm_in1>strm_in2
            case 1:
                result = 1;
                break; // strm_in1>=strm_in2
            case 2:
                result = 1;
                break; // strm_in1==strm_in2
            case 3:
                result = 0;
                break; // strm_in1!=strm_in2
            case 4:
                result = 1;
                break; // strm_in1<=strm_in2
            case 5:
                result = 0;
                break; // strm_in1<strm_in2
        }
    } else {
        switch (OP) {
            case 0:
                result = 0;
                break; // strm_in1>strm_in2
            case 1:
                result = 0;
                break; // strm_in1>=strm_in2
            case 2:
                result = 0;
                break; // strm_in1==strm_in2
            case 3:
                result = 1;
                break; // strm_in1!=strm_in2
            case 4:
                result = 1;
                break; // strm_in1<=strm_in2
            case 5:
                result = 1;
                break; // strm_in1<strm_in2
        }
    }
    return result;
}

// implement boolean algebra
inline bool alu_boolean(bool strm_in1, bool strm_in2, ap_uint<3> OP) {
#pragma HLS INLINE

    bool result;

    switch (OP) {
        case 0:
            result = strm_in2;
            break; // output stream2
        case 1:
            result = strm_in1;
            break; // output stream1
        case 2:
            result = true;
            break; // output constant 0
        case 3:
            result = false;
            break; // output constant 1
        case 4:
            result = strm_in1 && strm_in2;
            break; //&&
        case 5:
            result = strm_in1 || strm_in2;
            break; //||
        case 6:
            result = strm_in1 ^ strm_in2;
            break; // XOR
        case 7:
            result = strm_in1 == strm_in2;
            break; // XNOR
    }

    return result;
}

// implement multiplex
template <typename In_Type1, typename In_Type2, typename Out_Type>
Out_Type alu_mux(In_Type1 strm_in1, In_Type2 strm_in2, bool OP) {
#pragma HLS INLINE

    Out_Type result;

    if (OP) {
        result = strm_in1;
    } else {
        result = strm_in2;
    }

    return result;
}

// define cell result
template <typename Data_Type>
struct cell_data {
    Data_Type compute_result;
    bool boolean_result;
};

// implement alu_cell
template <typename In_Type1, typename In_Type2, typename Out_Type>
void alu_cell1(In_Type1 strm_in1, In_Type2 strm_in2, ap_uint<4> OP, cell_data<Out_Type>& cell_output) {
#pragma HLS INLINE

    Out_Type compute_result1;
    Out_Type compute_result2;
    bool boolean_result1;
    bool boolean_result2;

    compute_result1 = alu_mux<In_Type1, In_Type2, Out_Type>(strm_in1, strm_in2, OP(0, 0));
    boolean_result1 = alu_boolean(strm_in1, strm_in2, OP(2, 0));
    compute_result2 = alu_math<In_Type1, In_Type2, Out_Type>(strm_in1, strm_in2, OP(2, 0));
    boolean_result2 = alu_comparator<In_Type1, In_Type2>(strm_in1, strm_in2, OP(2, 0));

    if (OP(3, 3)) {
        cell_output.compute_result = compute_result1;
        cell_output.boolean_result = boolean_result1;
    } else {
        cell_output.compute_result = compute_result2;
        cell_output.boolean_result = boolean_result2;
    }
}

template <typename In_Type1, typename In_Type2, typename Out_Type>
void alu_cell2(cell_data<In_Type1> cell_Input1,
               cell_data<In_Type2> cell_Input2,
               ap_uint<4> OP,
               cell_data<Out_Type>& cell_output) {
#pragma HLS INLINE

    Out_Type compute_result1;
    Out_Type compute_result2;
    bool boolean_result1;
    bool boolean_result2;

    compute_result1 =
        alu_mux<In_Type1, In_Type2, Out_Type>(cell_Input1.compute_result, cell_Input2.compute_result, OP(0, 0));
    boolean_result1 = alu_boolean(cell_Input1.boolean_result, cell_Input2.boolean_result, OP(2, 0));
    compute_result2 =
        alu_math<In_Type1, In_Type2, Out_Type>(cell_Input1.compute_result, cell_Input2.compute_result, OP(2, 0));
    boolean_result2 =
        alu_comparator<In_Type1, In_Type2>(cell_Input1.compute_result, cell_Input2.compute_result, OP(2, 0));

    if (OP(3, 3)) {
        cell_output.compute_result = compute_result1;
        cell_output.boolean_result = boolean_result1;
    } else {
        cell_output.compute_result = compute_result2;
        cell_output.boolean_result = boolean_result2;
    }
}

// top module
template <typename TStrm1,
          typename TStrm2,
          typename TStrm3,
          typename TStrm4,

          typename TConst1,
          typename TConst2,
          typename TConst3,
          typename TConst4,

          typename TOut>
void dynamic_ALU_top(ap_uint<33> OP,

                     hls::stream<TStrm1>& strm_in1,
                     TConst1 c1,
                     hls::stream<TStrm2>& strm_in2,
                     TConst2 c2,
                     hls::stream<TStrm3>& strm_in3,
                     TConst3 c3,
                     hls::stream<TStrm4>& strm_in4,
                     TConst4 c4,
                     hls::stream<bool>& strm_in_end,

                     hls::stream<TOut>& strm_out,
                     hls::stream<bool>& strm_out_end) {
    TStrm1 in1;
    TStrm2 in2;
    TStrm3 in3;
    TStrm4 in4;

    cell_data<TOut> alu_cell1_result, alu_cell2_result, alu_cell3_result, alu_cell4_result, alu_cell5_result,
        alu_cell6_result, alu_cell7_result;

    TOut result;

    // assign operation
    bool output_mux;
    ap_uint<4> strm_empty, op1, op2, op3, op4, op5, op6, op7;

    output_mux = OP(32, 32);
    strm_empty = OP(31, 28);
    op1 = OP(27, 24);
    op2 = OP(23, 20);
    op3 = OP(19, 16);
    op4 = OP(15, 12);
    op5 = OP(11, 8);
    // op6 is determined by cell5 result
    op7 = OP(3, 0);

    bool end;
    // read the 1st end flag
    end = strm_in_end.read();

#ifndef __SYNTHESIS__
    int cnt = 0;
#endif

ALU_Processing:
    while (!end) {
#pragma HLS PIPELINE II = 1

#if 0
    // judge empty && read stream_in
    if (strm_empty(3, 3)) {
    } else {
      in1 = strm_in1.read();
    }

    if (strm_empty(2, 2)) {
    } else {
      in2 = strm_in2.read();
    }

    if (strm_empty(1, 1)) {
    } else {
      in3 = strm_in3.read();
    }

    if (strm_empty(0, 0)) {
    } else {
      in4 = strm_in4.read();
    }
#endif
        in1 = strm_in1.read();
        in2 = strm_in2.read();
        in3 = strm_in3.read();
        in4 = strm_in4.read();

#if !defined(__SYNTHESIS__) && _XFDB_DYN_EVAL_DEBUG == 1
        if (cnt < 10) {
            printf("in: %lld %lld %lld %lld, ", in1.to_int64(), in2.to_int64(), in3.to_int64(), in4.to_int64());
            printf("const: %lld %lld %lld %lld\n", c1.to_int64(), c2.to_int64(), c3.to_int64(), c4.to_int64());
        }
#endif

        end = strm_in_end.read();

        //-------------------------- cell level1 ----------------------------//

        // call cell 1-4 instance
        alu_cell1<TStrm1, TConst1, TOut>(in1, c1, op1, alu_cell1_result);

        alu_cell1<TStrm2, TConst2, TOut>(in2, c2, op2, alu_cell2_result);

        alu_cell1<TStrm3, TConst3, TOut>(in3, c3, op3, alu_cell3_result);

        alu_cell1<TStrm4, TConst4, TOut>(in4, c4, op4, alu_cell4_result);

#ifndef __SYNTHESIS__
#ifdef _XFDB_DYN_EVAL_DEBUG
        if (cnt < 10) {
            std::cout << "alu_cell1: op1=" << op1 << " in1=" << in1 << " c1=" << c1
                      << " result=" << alu_cell1_result.compute_result << std::endl;
            std::cout << "alu_cell1: op2=" << op2 << " in2=" << in2 << " c2=" << c2
                      << " result=" << alu_cell2_result.compute_result << std::endl;
            std::cout << "alu_cell1: op3=" << op3 << " in3=" << in3 << " c3=" << c3
                      << " result=" << alu_cell3_result.compute_result << std::endl;
            std::cout << "alu_cell1: op4=" << op4 << " in4=" << in4 << " c4=" << c4
                      << " result=" << alu_cell4_result.compute_result << std::endl;
        }
#endif
#endif

        //-------------------------- cell level2 ----------------------------//

        // call cell5
        alu_cell2<TOut, TOut, TOut>(alu_cell1_result, alu_cell2_result, op5, alu_cell5_result);

        // assign cell6 operation && call cell6
        op6(3, 1) = OP(7, 5);

        // when cell6 do multiplex, connect cell5 boolean result to cell6 as
        // multiplex signal
        if (op6(3, 3) == 1) {
            if (op6(1, 1) == 1) {
                // cell6 will choose multiplex
                op6(0, 0) = alu_cell5_result.boolean_result;
            } else {
                // cell6 will choose boolean_algbra
                op6(0, 0) = OP(4, 4);
            }
        } else {
            // cell6 will choose comparator or compute
            op6(0, 0) = OP(4, 4);
        }

        alu_cell2<TOut, TOut, TOut>(alu_cell3_result, alu_cell4_result, op6, alu_cell6_result);

#ifndef __SYNTHESIS__
#ifdef _XFDB_DYN_EVAL_DEBUG
        if (cnt < 10) {
            std::cout << "alu_cell2: op5=" << op5 << " result=" << alu_cell5_result.compute_result << std::endl;
            std::cout << "alu_cell2: op6=" << op6 << " result=" << alu_cell6_result.compute_result << std::endl;
        }
#endif
#endif

        //-------------------------- cell level3 ----------------------------//

        // call cell7 instance
        alu_cell2<TOut, TOut, TOut>(alu_cell5_result, alu_cell6_result, op7, alu_cell7_result);

        // choose result
        if (output_mux) {
            result = alu_cell7_result.boolean_result;
        } else {
            result = alu_cell7_result.compute_result;
        }

#if !defined(__SYNTHESIS__) && _XFDB_DYN_EVAL_DEBUG == 1
        if (cnt < 10) printf("out: %lld\n", result.to_int64());
        cnt++;
#endif

        // write output strm
        strm_out.write(result);
        strm_out_end.write(0);
    }
    // write end flag
    strm_out_end.write(1);
}

} // namespace details end

// clang-format off
/*
 * Operation description:
 *
 * Format:
 * |-------------------------------------Operator---------------------------------------|-----------User Define Operand---------|
 * |Output_Mux|Strm_Empty|Cell1_OP|Cell2_OP|Cell3_OP|Cell4_OP|Cell5_OP|Cell6_OP|Cell7_OP|Constant1|Constant2|Constant3|Constant4|
 * |---1bit---|---4bit---|--4bit--|--4bit--|--4bit--|--4bit--|--4bit--|--4bit--|--4bit--|--64bit--|--64bit--|--64bit--|--64bit--|
 *
 * Output_Mux: 0-->Output a compute result
 *             1-->Output a boolean result
 *
 * Strm_Empty: 0-->Strm not empty
 *             1-->Strm empty
 * For example: 0010-->Strm1, Strm2 and Strm4 are not empty and Strm3 is empty.
 *
 * Cell_OP: Four kind of cell operation which are
 * ALU_Math/ALU_Comparator/ALU_Boolean/ALU_Mux
 *
 *               A    B
 *                \  /
 *                Cell
 *                 |
 *               result
 *
 *     1.ALU_Math:
 *                  Add 0000-> A+B
 *                      0001-> A-B
 *                      0010-> -A+B
 *                      0011-> -A-B
 *                  Mul 0100-> A*B
 *                      0101-> -A*B
 *
 *     2.ALU_Comparator:
 *                      0000-> >>
 *                      0001-> >=
 *                      0010-> ==
 *                      0011-> !=
 *                      0100-> <=
 *                      0101-> <<
 *
 *     3.ALU_Mux:
 *                      1000-> select B
 *                      1001-> select A
 *                      1010-> select by internal boolean result
 *
 * (Only provide using cell5 boolean result to control multiplex select in cell6)
 *
 *     4.ALU_Boolean:
 *                      1000-> B
 *                      1001-> A
 *                      1010-> 0
 *                      1011-> 1
 *                      1100-> A && B
 *                      1101-> A || B
 *                      1110-> A XOR B
 *                      1111-> A XNOR B
 *
 * Example:
 *
 * 1. strm1*strm2
 *          Output_Mux: 0->Output a compute result
 *          Strm_Empty: 0011->Strm1, Strm2 are not empty while Strm3, Strm4 are empty
 *          cell1:      1001->select A
 *          cell2:      1001->select A
 *          cell3:      0000(any)
 *          cell4:      0000(any)
 *          cell5:      0100->A*B
 *          cell6:      0000(any)
 *          cell7:      1001->select A
 *          Operator:   0_0011_1001_1001_0000_0000_0100_0000_1001->0x039900409
 *          Operand:    0x0000000000000000
 *          Operation:  0x0399004090000000000000000
 *
 * 2. strm1*c1
 *          Output_Mux: 0->Output a compute result
 *          Strm_Empty: 0111->Strm1 is not empty while Strm2, Strm3, Strm4 are empty
 *          cell1:      0100->A*B
 *          cell2:      0000(any)
 *          cell3:      0000(any)
 *          cell4:      0000(any)
 *          cell5:      1001->select A
 *          cell6:      0000(any)
 *          cell7:      1001->select A
 *          Operator:   0_0111_0100_0000_0000_0000_1001_0000_1001->0x074000909
 *          Operand:    user define
 *          Operation:  0x074900909/user defined operand
 *
 * 3. strm1*(-strm2+c2)
 *          Output_Mux: 0->Output a compute result
 *          Strm_Empty: 0011->Strm1, Strm2 is not empty while Strm3, Strm4 are empty
 *          cell1:      1001->select A
 *          cell2:      0010->-A+B
 *          cell3:      0000(any)
 *          cell4:      0000(any)
 *          cell5:      0100->A*B
 *          cell6:      0000(any)
 *          cell7:      1001->select A
 *          Operator:   0_0011_1001_0010_0000_0000_0100_0000_1001->0x035200409
 *          Operand:    user define
 *          Operation:  0x035200409/user defined operand
 *
 * 4. strm1*(-strm2+c2)-strm3*strm4
 *          Output_Mux: 0->Output a compute result
 *          Strm_Empty: 0000->Strm1, Strm2, Strm3, Strm4 are not empty
 *          cell1:      1001->select A
 *          cell2:      0010->-A+B
 *          cell3:      1001->select A
 *          cell4:      1001->select A
 *          cell5:      0100->A*B
 *          cell6:      0100->A*B
 *          cell7:      0001->A-B
 *          Operator:   0_0000_1001_0010_1001_1001_0100_0100_0001->0x009299441
 *          Operand:    user define
 *          Operation:  0x009299441/user defined operand
 *
 * 5. strm1*(-strm2+c2)*(strm3+c3)
 *          Output_Mux: 0->Output a compute result
 *          Strm_Empty: 0001->Strm1, Strm2, Strm3 is not empty while Strm4 is empty
 *          cell1:      1001->select A
 *          cell2:      0010->-A+B
 *          cell3:      0000->A+B
 *          cell4:      0000(any)
 *          cell5:      0100->A*B
 *          cell6:      1001->select A
 *          cell7:      0100->A*B
 *          Operator:   0_0001_1001_0010_0000_0000_0100_1001_0100->0x019200494
 *          Operand:    user define
 *          Operation:  0x019200494/user defined operand
 *
 * 6. strm1==strm2 ? strm3:c4
 *          Output_Mux:    0->Output a compute result
 *          Strm_Empty: 0001->Strm1, Strm2, Strm3 is not empty while Strm4 is empty
 *          cell1:      1001->select A
 *          cell2:      1001->select A
 *          cell3:      1001->select A
 *          cell4:      1000->select B
 *          cell5:      0010->A==B
 *          cell6:      1010->select by cell5 boolean result
 *          cell7:      1000->select B
 *          Operator:   0_0001_1001_1001_1001_1000_0010_1010_1000->0x0199982A8
 *          Operand:    user define
 *          Operation:  0x0199982A8/user defined operand
 *
 * 7. ((strm1==c1) || (strm2==c2)) ? c3:c4
 *          Output_Mux: 0->Output a compute result
 *          Strm_Empty: 0011->Strm1, Strm2 is not empty while Strm3, Strm4 are empty
 *          cell1:      0010->A==B
 *          cell2:      0010->A==B
 *          cell3:      1000->select B
 *          cell4:      1000->select B
 *          cell5:      1000->A || B
 *          cell6:      1010->select by cell5 boolean result
 *          cell7:      1000->select B
 *          Operator:   0_0011_0010_0010_1000_1000_1101_1010_1000->0x0322888A8
 *          Operand:    user define
 *          Operation:  0x032288DA8/user defined operand
 *
 * 8. ((strm1!=c1) && (strm2!=c2)) ? c3:c4
 *          Output_Mux: 0->Output a compute result
 *          Strm_Empty: 0011->Strm1, Strm2 is not empty while Strm3, Strm4 are empty
 *          cell1:      0011->A!=B
 *          cell2:      0011->A!=B
 *          cell3:      1000->select B
 *          cell4:      1000->select B
 *          cell5:      1001->A && B
 *          cell6:      1010->select by cell5 boolean result
 *          cell7:      1000->select B
 *          Operator:   0_0011_0011_0011_1000_1000_1100_1010_1000->0x0333889A8
 *          Operand:    user define
 *          Operation:  0x033388CA8/user defined operand
 *
 *
 * Other Example Operators:
 *
 *-----------------------one strm------------------------------------
 * 1.strm1+c1: 0x070000909
 *
 * 2.strm1>c1: 0x170000909
 *
 * 3.strm1&&c1: 0x17C000909
 *
 * ------------------------two strm------------------------------------
 * 1.strm1+strm2:0x039900009
 *
 * 2.strm1>strm2: 0x139900009
 *
 * 3.strm1==strm2: 0x139900209
 *
 * 4.(strm1+c1)*(strm2+c2):0x030000409
 *
 * ------------------------three strm----------------------------------
 * 1.strm1+strm2+strm3:
 *
 * 2.strm1*strm2*strm3:
 *
 * 3.(strm1+c1)*(strm2+c2)*(strm3+c3): 0x010000494
 *
 * -------------------------four strm----------------------------------
 * 1.(strm1+strm2)+(strm3+strm4): 0x09999000
 *
 * 2.((strm1+c1)*(strm2+c2))*((strm3+c3)*(strm4+c4)): 0x04440000
 *
 */
// clang-format on

/**
 * @brief Dynamic expression evaluation.
 *
 * This primitive has four fixed number of column inputs, and allows up to four constants to be specified via
 * configuration. The operation between the column values and constants can be defined dynamically through the
 * configuration at run-time. The same configuration is used for all rows until the end of input.
 *
 * The constant numbers are assumed to be no more than 32-bits.
 *
 * For the definition of the config word, please refer to the "Design Internal" Section of the document and the
 * corresponding test in ``L1/tests``.
 *
 * @tparam TStrm1 Type of input Stream1
 * @tparam TStrm2 Type of input Stream2
 * @tparam TStrm3 Type of input Stream3
 * @tparam TStrm4 Type of input Stream4
 *
 * @tparam TConst1 Type of input Constant1
 * @tparam TConst2 Type of input Constant2
 * @tparam TConst3 Type of input Constant3
 * @tparam TConst4 Type of input Constant4
 *
 * @tparam TOut Type of Compute Result
 *
 * @param config configuration bits of ops and constants.
 *
 * @param strm_in1 input Stream1
 * @param strm_in2 input Stream2
 * @param strm_in3 input Stream3
 * @param strm_in4 input Stream4
 * @param strm_in_end end flag of input stream
 *
 * @param strm_out output Stream
 * @param strm_out_end end flag of output stream
 */
template <typename TStrm1,
          typename TStrm2,
          typename TStrm3,
          typename TStrm4,

          typename TConst1,
          typename TConst2,
          typename TConst3,
          typename TConst4,

          typename TOut>
void dynamicEval(ap_uint<289> config,

                 hls::stream<TStrm1>& strm_in1,
                 hls::stream<TStrm2>& strm_in2,
                 hls::stream<TStrm3>& strm_in3,
                 hls::stream<TStrm4>& strm_in4,
                 hls::stream<bool>& strm_in_end,

                 hls::stream<TOut>& strm_out,
                 hls::stream<bool>& strm_out_end) {
#pragma HLS INLINE

    ap_uint<33> Operator;
    Operator = config(288, 256);

    TConst1 c1 = config(255, 192);
    TConst2 c2 = config(191, 128);
    TConst3 c3 = config(127, 64);
    TConst4 c4 = config(63, 0);

    details::dynamic_ALU_top<TStrm1, TStrm2, TStrm3, TStrm4, TConst1, TConst2, TConst3, TConst4, TOut>(
        Operator,

        strm_in1, c1, strm_in2, c2, strm_in3, c3, strm_in4, c4, strm_in_end,

        strm_out, strm_out_end);
}

} // namespace database end
} // namespace xf end

#endif // XF_DATABASE_DYNAMIC_EVAL_H
