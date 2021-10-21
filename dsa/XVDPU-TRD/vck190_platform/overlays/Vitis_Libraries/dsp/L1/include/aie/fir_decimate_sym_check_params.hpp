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
#ifndef _DSPLIB_FIR_DECIMATE_SYM_CHECK_PARAMS_HPP_
#define _DSPLIB_FIR_DECIMATE_SYM_CHECK_PARAMS_HPP_

// This file holds the static_assert statements which alert the user to any illegal or unsupported
// values and combinations of values of template parameters for the library element in question.
#include "fir_utils.hpp"
#include "fir_decimate_sym_traits.hpp"

// Parameter value defensive and legality checks
static_assert(TP_DECIMATE_FACTOR <= fnMaxDecimateFactor<TT_DATA, TT_COEFF>(),
              "ERROR: Max Decimate factor exxceeded. High Decimate factors do not take advantage from symmetrical "
              "implementation. Use fir_decimate_asym instead.");
static_assert(TP_FIR_LEN <= FIR_LEN_MAX, "ERROR: Max supported FIR length exceeded. ");
static_assert(TP_FIR_RANGE_LEN >= FIR_LEN_MIN,
              "ERROR: Illegal combination of design FIR length and cascade length, resulting in kernel FIR length "
              "below minimum required value. ");
static_assert(TP_FIR_LEN % TP_DECIMATE_FACTOR == 0, "ERROR: TP_FIR_LEN must be a multiple of TP_DECIMATE_FACTOR");
static_assert(TP_SHIFT >= SHIFT_MIN && TP_SHIFT <= SHIFT_MAX, "ERROR: TP_SHIFT is out of the supported range.");
static_assert(TP_RND >= ROUND_MIN && TP_RND <= ROUND_MAX, "ERROR: TP_RND is out of the supported range.");
static_assert(fnEnumType<TT_DATA>() != enumUnknownType, "ERROR: TT_DATA is not a supported type.");
static_assert(fnEnumType<TT_COEFF>() != enumUnknownType, "ERROR: TT_COEFF is not a supported type.");
static_assert(fnFirDecSymTypeSupport<TT_DATA, TT_COEFF>() != 0,
              "ERROR: The combination of TT_DATA and TT_COEFF is not supported for this class.");
static_assert(fnFirDecSymmertySupported<TT_DATA, TT_COEFF>() != 0,
              "ERROR: The combination of TT_DATA and TT_COEFF is not supported for this class, as implementation would "
              "not use the benefits of symmetry. Use fir_decimate_asym instead.");
static_assert(fnTypeCheckDataCoeffSize<TT_DATA, TT_COEFF>() != 0,
              "ERROR: TT_DATA type less precise than TT_COEFF is not supported.");
static_assert(fnTypeCheckDataCoeffCmplx<TT_DATA, TT_COEFF>() != 0,
              "ERROR: real TT_DATA with complex TT_COEFF is not supported.");
static_assert(fnTypeCheckDataCoeffFltInt<TT_DATA, TT_COEFF>() != 0,
              "ERROR: a mix of float and integer types of TT_DATA and TT_COEFF is not supported.");
static_assert(fnFirDecSymMultiColumn<TT_DATA, TT_COEFF>() != 0,
              "ERROR: The combination of TT_DATA and TT_COEFF is currently unsupported.");
#endif // _DSPLIB_FIR_DECIMATE_SYM_CHECK_PARAMS_HPP_
