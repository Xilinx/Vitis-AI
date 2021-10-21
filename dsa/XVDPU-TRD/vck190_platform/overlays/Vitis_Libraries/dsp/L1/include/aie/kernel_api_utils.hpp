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
#pragma once

#ifndef _DSPLIB_RELOAD_UTILS_HPP_
#define _DSPLIB_RELOAD_UTILS_HPP_

// This file holds sets of templated types and overloaded (or template specialized) functions
// for use by multiple kernels.
// Functions in this file as a rule use intrinsics from a single set. For instance, a set
// may contain a MAC with pre-add which uses a single 1024 bit buffer for both forward
// and reverse data. In cases where a library element has to use an intrinsic which differs
// by more than the types used for some combinations of library element parameter types
// then the set of templatized functions will be particular to that library element and should
// therefore be in <library_element>_utils.hpp

#include <stdio.h>
#include <adf.h>
#include "fir_utils.hpp"

namespace xf {
namespace dsp {
namespace aie {

// ---------------------------------------------------- AIE::API ---------------------------------------------------- //

// Unroll functions. Ported from AIE::API for independent use.
template <typename T, T Start, T End, T It>
struct unroll_context {
    constexpr operator T() const { return current(); }

    constexpr T min() const { return Start; }

    constexpr T max() const { return End; }

    constexpr T current() const { return It; }
};

template <typename T, T Start, T End, T It, T Step>
struct unroll_for_helper {
    static_assert(Step != 0, "0 is not a valid step");

    template <typename Fn>
    void execute(Fn&& fn) {
        if
            constexpr(It == End) return;

        constexpr unroll_context<T, Start, End, It> ctx{};

        static_assert(std::is_invocable_v<Fn, decltype(ctx)> || std::is_invocable_v<Fn>);

        if
            constexpr(std::is_invocable_v<Fn, decltype(ctx)>) fn(ctx);
        else
            fn();

        constexpr T next_it = It + Step;

        // Check for wrap-around
        constexpr bool is_next_it_valid =
            ((Step > 0) && (next_it > It) && (next_it < End)) || ((Step < 0) && (next_it < It) && (next_it > End));

        if
            constexpr(is_next_it_valid) unroll_for_helper<T, Start, End, next_it, Step>().execute(std::forward<Fn>(fn));
    }
};

template <typename T, T Start, T End, T Step = 1, typename Fn = void>
void unroll_for(Fn&& fn) {
    unroll_for_helper<T, Start, End, Start, Step>().execute(std::forward<Fn>(fn));
}

template <unsigned Times, typename Fn>
void unroll_times(Fn&& fn) {
    unroll_for<unsigned, 0, Times>(std::forward<Fn>(fn));
}

// function to return ::aie::detail:AccumClass enum based on input data type
template <typename TT_DATA>
inline constexpr::aie::detail::AccumClass fnAccClass() {
    return ::aie::detail::AccumClass::Int; // ::aie::AccumClass -no need to go to detail
};
template <>
inline constexpr::aie::detail::AccumClass fnAccClass<int16>() {
    return ::aie::detail::AccumClass::Int;
};
template <>
inline constexpr::aie::detail::AccumClass fnAccClass<cint16>() {
    return ::aie::detail::AccumClass::CInt;
};
template <>
inline constexpr::aie::detail::AccumClass fnAccClass<int32>() {
    return ::aie::detail::AccumClass::Int;
};
template <>
inline constexpr::aie::detail::AccumClass fnAccClass<cint32>() {
    return ::aie::detail::AccumClass::CInt;
};
template <>
inline constexpr::aie::detail::AccumClass fnAccClass<float>() {
    return ::aie::detail::AccumClass::FP;
};
template <>
inline constexpr::aie::detail::AccumClass fnAccClass<cfloat>() {
    return ::aie::detail::AccumClass::CFP;
};

// function to return ::aie::detail:AccumClass enum based on input data type
template < ::aie::detail::AccumClass Accum, unsigned Size>
struct accClassTag {};
template <>
struct accClassTag< ::aie::detail::AccumClass::Int, 48> {
    using type = acc48;
};
template <>
struct accClassTag< ::aie::detail::AccumClass::Int, 80> {
    using type = acc80;
};
template <>
struct accClassTag< ::aie::detail::AccumClass::CInt, 48> {
    using type = cacc48;
};
template <>
struct accClassTag< ::aie::detail::AccumClass::CInt, 80> {
    using type = cacc80;
};
template <>
struct accClassTag< ::aie::detail::AccumClass::FP, 32> {
    using type = accfloat;
};
template <>
struct accClassTag< ::aie::detail::AccumClass::CFP, 32> {
    using type = caccfloat;
};

template < ::aie::detail::AccumClass Acc, unsigned Size>
using accClassTag_t = typename accClassTag<Acc, Size>::type;

// T_buff structs with ::aie::vectors
template <typename T>
struct T_buff_128b {
    using v_type = ::aie::vector<T, 128 / 8 / sizeof(T)>;
    v_type val;
};
template <typename T>
struct T_buff_256b {
    using v_type = ::aie::vector<T, 256 / 8 / sizeof(T)>;
    v_type val;
};
template <typename T>
struct T_buff_512b {
    using v_type = ::aie::vector<T, 512 / 8 / sizeof(T)>;
    v_type val;
};
template <typename T>
struct T_buff_1024b {
    using v_type = ::aie::vector<T, 1024 / 8 / sizeof(T)>;
    v_type val;
};

// function to window_readincr a 256 bit vector
template <typename TT_DATA>
T_buff_256b<TT_DATA> inline window_readincr_256b(input_window<TT_DATA>* inWindow) {
    T_buff_256b<TT_DATA> result;
    result.val = window_readincr_v<256 / 8 / sizeof(TT_DATA)>(inWindow);
    return result;
};
// function to window_readincr a 256 bit vector
template <typename TT_DATA>
T_buff_128b<TT_DATA> inline window_readincr_128b(input_window<TT_DATA>* inWindow) {
    T_buff_128b<TT_DATA> result;
    result.val = window_readincr_v<128 / 8 / sizeof(TT_DATA)>(inWindow);
    return result;
};

// function to window_readincr a 256 bit vector
template <typename TT_DATA>
T_buff_256b<TT_DATA> inline window_readdecr_256b(input_window<TT_DATA>* inWindow) {
    T_buff_256b<TT_DATA> result;
    result.val = window_readdecr_v<256 / 8 / sizeof(TT_DATA)>(inWindow);
    return result;
};
// function to window_readincr a 256 bit vector
template <typename TT_DATA>
T_buff_128b<TT_DATA> inline window_readdecr_128b(input_window<TT_DATA>* inWindow) {
    T_buff_128b<TT_DATA> result;
    result.val = window_readdecr_v<128 / 8 / sizeof(TT_DATA)>(inWindow);
    return result;
};

// function to readincr a 128 bit vector from a stream
template <typename TT_DATA>
T_buff_128b<TT_DATA> inline stream_readincr_128b(input_stream<TT_DATA>* inStream) {
    T_buff_128b<TT_DATA> result;
    result.val = readincr_v<128 / 8 / sizeof(TT_DATA)>(inStream);
    return result;
};

// function to writeincr a 128 bit vector to a stream
template <typename TT_DATA>
void inline stream_writeincr_128b(output_stream<TT_DATA>* outStream, T_buff_128b<TT_DATA> data) {
    writeincr_v<128 / 8 / sizeof(TT_DATA)>(outStream);
};

// Update buffer with 256-bit read from the input window with pointer increment.
template <typename TT_DATA>
inline void upd_win_incr_256b(T_buff_1024b<TT_DATA>& xbuff, int index, input_window<TT_DATA>* inWindow) {
    xbuff.val.insert(index % 4, window_readincr_v<256 / 8 / sizeof(TT_DATA)>(
                                    inWindow)); // ::aie::api version of upd_w with window decrement
};
// Update buffer with 256-bit read from the input window with pointer decrement.
template <typename TT_DATA>
inline void upd_win_decr_256b(T_buff_1024b<TT_DATA>& xbuff, int index, input_window<TT_DATA>* inWindow) {
    xbuff.val.insert(index % 4, window_readdecr_v<256 / 8 / sizeof(TT_DATA)>(
                                    inWindow)); // ::aie::api version of upd_w with window decrement
};

// Update buffer with 128-bit read from the input window with pointer increment.
template <typename TT_DATA>
inline void upd_win_incr_128b(T_buff_1024b<TT_DATA>& xbuff, int index, input_window<TT_DATA>* inWindow) {
    xbuff.val.insert(index % 8, window_readincr_v<128 / 8 / sizeof(TT_DATA)>(
                                    inWindow)); // ::aie::api version of upd_w with window decrement
};
// Update buffer with 256-bit read from the input window with pointer decrement.
template <typename TT_DATA>
inline void upd_win_decr_128b(T_buff_1024b<TT_DATA>& xbuff, int index, input_window<TT_DATA>* inWindow) {
    xbuff.val.insert(index % 8, window_readdecr_v<128 / 8 / sizeof(TT_DATA)>(
                                    inWindow)); // ::aie::api version of upd_w with window decrement
};

// function to clear buffer
template <typename TT_DATA>
T_buff_1024b<TT_DATA> inline null_buff_1024b() {
    T_buff_1024b<TT_DATA> ret;
    ret.val = ::aie::zeros<TT_DATA, ret.val.size()>();
    return ret;
};

// function to clear 512b buffer
template <typename TT_DATA>
T_buff_512b<TT_DATA> inline null_buff_512b() {
    T_buff_512b<TT_DATA> ret;
    ret.val = ::aie::zeros<TT_DATA, ret.val.size()>();
    return ret;
};

// function to clear 256b buffer
template <typename TT_DATA>
T_buff_256b<TT_DATA> inline null_buff_256b() {
    T_buff_256b<TT_DATA> ret;
    ret.val = ::aie::zeros<TT_DATA, ret.val.size()>();
    return ret;
};

inline namespace fir_api {
// T_acc struct with ::aie::accum
template <typename TT_DATA, typename TT_COEFF>
struct T_acc {
    using v_type =
        ::aie::detail::accum<fnAccClass<TT_DATA>(), fnAccSize<TT_DATA, TT_COEFF>(), fnNumLanes<TT_DATA, TT_COEFF>()>;
    v_type val, uval;
};

// T_acc384 struct with ::aie::accum
template <typename TT_DATA, typename TT_COEFF>
struct T_acc384 {
    using v_type =
        ::aie::detail::accum<fnAccClass<TT_DATA>(), fnAccSize<TT_DATA, TT_COEFF>(), fnNumLanes384<TT_DATA, TT_COEFF>()>;
    v_type val, uval;
};

// T_outVal struct with ::aie::vector
template <typename TT_DATA, typename TT_COEFF>
struct T_outVal {
    using v_type = ::aie::vector<TT_DATA, fnNumLanes<TT_DATA, TT_COEFF>()>;
    v_type val;
};
// T_outVal384 struct with ::aie::vector
template <typename TT_DATA, typename TT_COEFF>
struct T_outVal384 {
    using v_type = ::aie::vector<TT_DATA, fnNumLanes384<TT_DATA, TT_COEFF>()>;
    v_type val;
};

// Shift and saturate call to allow null operation
template <typename TT_DATA, typename TT_COEFF>
inline T_outVal<TT_DATA, TT_COEFF> shiftAndSaturate(T_acc<TT_DATA, TT_COEFF> acc, int shift) {
    T_outVal<TT_DATA, TT_COEFF> retVal;
    retVal.val = acc.val.template to_vector<TT_DATA>(shift);
    return retVal;
}
// Shift and saturate call to allow null operation
template <typename TT_DATA, typename TT_COEFF>
inline T_outVal384<TT_DATA, TT_COEFF> shiftAndSaturate(T_acc384<TT_DATA, TT_COEFF> acc, int shift) {
    T_outVal384<TT_DATA, TT_COEFF> retVal;
    retVal.val = acc.val.template to_vector<TT_DATA>(shift);
    return retVal;
}

template <bool T_CASC_IN, typename T_D, unsigned int T_DUAL_IP = 0>
using T_inputIF = fir::T_inputIF<T_CASC_IN, T_D, T_DUAL_IP>;
template <bool T_CASC_IN, typename T_D>
using T_outputIF = fir::T_outputIF<T_CASC_IN, T_D>;

// Overloaded function to write to window output.
template <typename TT_DATA, typename TT_COEFF, unsigned int TP_NUM_OUTPUTS = 1>
inline void writeWindow(T_outputIF<CASC_OUT_TRUE, TT_DATA> outInterface, T_outVal<TT_DATA, TT_COEFF> outVal) {
    // Do nothing if cascade output is present, but window output is not
}

template <typename TT_DATA, typename TT_COEFF, unsigned int TP_NUM_OUTPUTS = 1>
inline void writeWindow(T_outputIF<CASC_OUT_FALSE, TT_DATA> outInterface, T_outVal<TT_DATA, TT_COEFF> outVal) {
    window_writeincr(outInterface.outWindow, outVal.val);
    if
        constexpr(TP_NUM_OUTPUTS == 2) { window_writeincr(outInterface.outWindow2, outVal.val); }
}

// Overloaded function to write to window output.
template <typename TT_DATA, typename TT_COEFF, unsigned int TP_NUM_OUTPUTS = 1>
inline void writeWindow(T_outputIF<CASC_OUT_TRUE, TT_DATA> outInterface, T_outVal384<TT_DATA, TT_COEFF> outVal) {
    // Do nothing if cascade output is present, but window output is not
}

template <typename TT_DATA, typename TT_COEFF, unsigned int TP_NUM_OUTPUTS = 1>
inline void writeWindow(T_outputIF<CASC_OUT_FALSE, TT_DATA> outInterface, T_outVal384<TT_DATA, TT_COEFF> outVal) {
    window_writeincr(outInterface.outWindow, outVal.val);
    if
        constexpr(TP_NUM_OUTPUTS == 2) { window_writeincr(outInterface.outWindow2, outVal.val); }
}

// Overloaded function to skip writing to cascade output.
template <typename TT_DATA, typename TT_COEFF>
inline void writeCascade(T_outputIF<CASC_OUT_FALSE, TT_DATA> outInterface, T_acc<TT_DATA, TT_COEFF> acc) {
    // Do nothing if window output is present, but cascade output is not
}

// Overloaded function to write to cascade output.
template <typename TT_DATA, typename TT_COEFF>
inline void writeCascade(T_outputIF<CASC_OUT_TRUE, TT_DATA> outInterface, T_acc<TT_DATA, TT_COEFF> acc) {
    // Supports complex 48-bit only. Other types require cast
    using accTag = accClassTag_t<fnAccClass<TT_DATA>(), fnAccSize<TT_DATA, TT_COEFF>()>;
    writeincr<accTag, fnNumLanes<TT_DATA, TT_COEFF>()>((output_stream<accTag>*)outInterface.outCascade, acc.val);
}

// Overloaded function to write to cascade output.
template <typename TT_DATA, typename TT_COEFF>
inline void writeCascade(T_outputIF<CASC_OUT_FALSE, TT_DATA> outInterface, T_acc384<TT_DATA, TT_COEFF> acc) {
    // Do nothing
}

// Overloaded function to write to cascade output.
template <typename TT_DATA, typename TT_COEFF, unsigned int TP_DUAL_IP = 0>
inline void writeCascade(T_outputIF<CASC_OUT_TRUE, TT_DATA> outInterface, T_acc384<TT_DATA, TT_COEFF> acc) {
    using accTag = accClassTag_t<fnAccClass<TT_DATA>(), fnAccSize<TT_DATA, TT_COEFF>()>;
    writeincr<accTag, fnNumLanes384<TT_DATA, TT_COEFF>()>((output_stream<accTag>*)outInterface.outCascade, acc.val);
}

// Overloaded function to read from cascade input.
template <typename TT_DATA, typename TT_COEFF, unsigned int TP_DUAL_IP = 0>
inline T_acc<TT_DATA, TT_COEFF> readCascade(T_inputIF<false, TT_DATA, TP_DUAL_IP> inInterface,
                                            T_acc<TT_DATA, TT_COEFF> acc) {
    // Do nothing
    T_acc<TT_DATA, TT_COEFF> ret;
    using accTag = accClassTag_t<fnAccClass<TT_DATA>(), fnAccSize<TT_DATA, TT_COEFF>()>;
    ret.val = ::aie::zeros<accTag, ret.val.size()>();
    return ret;
};

// //Overloaded function to read from cascade input.
template <typename TT_DATA, typename TT_COEFF, unsigned int TP_DUAL_IP = 0>
inline T_acc<TT_DATA, TT_COEFF> readCascade(T_inputIF<true, TT_DATA, TP_DUAL_IP> inInterface,
                                            T_acc<TT_DATA, TT_COEFF> acc) {
    // Supports complex only. Real types require cast
    T_acc<TT_DATA, TT_COEFF> ret;
    using accTag = accClassTag_t<fnAccClass<TT_DATA>(), fnAccSize<TT_DATA, TT_COEFF>()>;
    ret.val = readincr_v<fnNumLanes<TT_DATA, TT_COEFF>(), accTag>((input_stream<accTag>*)inInterface.inCascade);
    return ret;
};

// Overloaded function to read from cascade input.
template <typename TT_DATA, typename TT_COEFF, unsigned int TP_DUAL_IP = 0>
inline T_acc384<TT_DATA, TT_COEFF> readCascade(T_inputIF<false, TT_DATA, TP_DUAL_IP> inInterface,
                                               T_acc384<TT_DATA, TT_COEFF> acc) {
    // Do nothing
    T_acc384<TT_DATA, TT_COEFF> ret;
    using accTag = accClassTag_t<fnAccClass<TT_DATA>(), fnAccSize<TT_DATA, TT_COEFF>()>;
    ret.val = ::aie::zeros<accTag, ret.val.size()>();
    return ret;
};

// //Overloaded function to read from cascade input.
template <typename TT_DATA, typename TT_COEFF, unsigned int TP_DUAL_IP = 0>
inline T_acc384<TT_DATA, TT_COEFF> readCascade(T_inputIF<true, TT_DATA, TP_DUAL_IP> inInterface,
                                               T_acc384<TT_DATA, TT_COEFF> acc) {
    // Supports complex only. Real types require cast
    T_acc384<TT_DATA, TT_COEFF> ret;
    using accTag = accClassTag_t<fnAccClass<TT_DATA>(), fnAccSize<TT_DATA, TT_COEFF>()>;
    ret.val = readincr_v<fnNumLanes384<TT_DATA, TT_COEFF>(), accTag>((input_stream<accTag>*)inInterface.inCascade);
    return ret;
};
}
}
}
} // namespaces

#include "kernel_rtp_reload.hpp"
#include "kernel_broadcast.hpp"

#endif // _DSPLIB_RELOAD_UTILS_HPP_
