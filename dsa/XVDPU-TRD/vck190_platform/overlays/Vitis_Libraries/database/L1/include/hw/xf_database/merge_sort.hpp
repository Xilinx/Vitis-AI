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

#ifndef _XF_DATABASE_MERGE_SORT_HPP_
#define _XF_DATABASE_MERGE_SORT_HPP_

#include <ap_int.h>
#include <hls_stream.h>

namespace xf {
namespace database {
namespace details {

template <typename Data_Type, typename Key_Type>
void merge_sort_top(hls::stream<Data_Type>& left_din_strm,
                    hls::stream<Key_Type>& left_kin_strm,
                    hls::stream<bool>& left_strm_in_end,

                    hls::stream<Data_Type>& right_din_strm,
                    hls::stream<Key_Type>& right_kin_strm,
                    hls::stream<bool>& right_strm_in_end,

                    hls::stream<Data_Type>& dout_strm,
                    hls::stream<Key_Type>& kout_strm,
                    hls::stream<bool>& strm_out_end,

                    bool sign) {
    Key_Type iLeft;
    Key_Type iRight;
    Key_Type Result;
    Data_Type dLeft;
    Data_Type dRight;
    Data_Type dResult;

    bool left_strm_end = 0;
    bool right_strm_end = 0;

    // control output the last element
    bool output_last_left = 0;
    bool output_last_right = 0;

    // read 1st left key
    left_strm_end = left_strm_in_end.read();
    if (!left_strm_end) {
        iLeft = left_kin_strm.read();
        dLeft = left_din_strm.read();
        left_strm_end = left_strm_in_end.read();
    }

    // read 1st right key
    right_strm_end = right_strm_in_end.read();
    if (!right_strm_end) {
        iRight = right_kin_strm.read();
        dRight = right_din_strm.read();
        right_strm_end = right_strm_in_end.read();
    }

merge_loop:
    while (!left_strm_end || !right_strm_end) {
#pragma HLS PIPELINE II = 1

        // output && read next left or right
        if (sign) {
            //----------------------------------ascending-------------------------------//
            if (iLeft < iRight) {
                if (!left_strm_end) {
                    // output left && read next left
                    Result = iLeft;
                    dResult = dLeft;

                    iLeft = left_kin_strm.read();
                    dLeft = left_din_strm.read();
                    left_strm_end = left_strm_in_end.read();
                } else if (left_strm_end && !output_last_left && !right_strm_end) {
                    // output last left && no read
                    Result = iLeft;
                    dResult = dLeft;
                    output_last_left = 1;
                } else if (left_strm_end && output_last_left && !right_strm_end) {
                    // output right && read risidual right
                    Result = iRight;
                    dResult = dRight;

                    iRight = right_kin_strm.read();
                    dRight = right_din_strm.read();
                    right_strm_end = right_strm_in_end.read();
                } else {
                    // left_strm_end && right_strm_end
                    // no operation, break loop
                }
            } else {
                if (!right_strm_end) {
                    // output right && read next right
                    Result = iRight;
                    dResult = dRight;

                    iRight = right_kin_strm.read();
                    dRight = right_din_strm.read();
                    right_strm_end = right_strm_in_end.read();
                } else if (right_strm_end && !output_last_right && !left_strm_end) {
                    // output last right && no read
                    Result = iRight;
                    dResult = dRight;
                    output_last_right = 1;
                } else if (right_strm_end && output_last_right && !left_strm_end) {
                    // output left && read risidual left
                    Result = iLeft;
                    dResult = dLeft;

                    iLeft = left_kin_strm.read();
                    dLeft = left_din_strm.read();
                    left_strm_end = left_strm_in_end.read();
                } else {
                    // left_strm_end && right_strm_end
                    // no operation, break loop
                }
            }
            //-------------------------------------end---------------------------------//
        } else {
            //----------------------------------descending-----------------------------//
            if (iLeft < iRight) {
                if (!right_strm_end) {
                    // output right && read next right
                    Result = iRight;
                    dResult = dRight;

                    iRight = right_kin_strm.read();
                    dRight = right_din_strm.read();
                    right_strm_end = right_strm_in_end.read();
                } else if (right_strm_end && !output_last_right && !left_strm_end) {
                    // output last right && no read
                    Result = iRight;
                    dResult = dRight;
                    output_last_right = 1;
                } else if (right_strm_end && output_last_right && !left_strm_end) {
                    // output left && read risidual left
                    Result = iLeft;
                    dResult = dLeft;

                    iLeft = left_kin_strm.read();
                    dLeft = left_din_strm.read();
                    left_strm_end = left_strm_in_end.read();
                } else {
                    // left_strm_end && right_strm_end
                    // no operation, break loop
                }
            } else {
                if (!left_strm_end) {
                    // output left && read next left
                    Result = iLeft;
                    dResult = dLeft;

                    iLeft = left_kin_strm.read();
                    dLeft = left_din_strm.read();
                    left_strm_end = left_strm_in_end.read();
                } else if (left_strm_end && !output_last_left && !right_strm_end) {
                    // output last left && no read
                    Result = iLeft;
                    dResult = dLeft;
                    output_last_left = 1;
                } else if (left_strm_end && output_last_left && !right_strm_end) {
                    // output right && read risidual right
                    Result = iRight;
                    dResult = dRight;

                    iRight = right_kin_strm.read();
                    dRight = right_din_strm.read();
                    right_strm_end = right_strm_in_end.read();
                } else {
                    // left_strm_end && right_strm_end
                    // no operation, break loop
                }
            }
            //-------------------------------------end---------------------------------//
        }
        kout_strm.write(Result);
        dout_strm.write(dResult);
        strm_out_end.write(0);
    }

    //------------------------------output last element------------------------//
    if (sign) {
        if (iLeft < iRight) {
            // whether output last left
            if (!output_last_left) {
                kout_strm.write(iLeft);
                dout_strm.write(dLeft);
                strm_out_end.write(0);
            } else {
                // no operation
            }
            // whether output last right
            if (!output_last_right) {
                kout_strm.write(iRight);
                dout_strm.write(dRight);
                strm_out_end.write(0);
            } else {
                // no operation
            }
        } else {
            // whether output last right
            if (!output_last_right) {
                kout_strm.write(iRight);
                dout_strm.write(dRight);
                strm_out_end.write(0);
            } else {
                // no operation
            }
            // whether output last left
            if (!output_last_left) {
                kout_strm.write(iLeft);
                dout_strm.write(dLeft);
                strm_out_end.write(0);
            } else {
                // no operation
            }
        }
    } else {
        if (iLeft < iRight) {
            // whether output last right
            if (!output_last_right) {
                kout_strm.write(iRight);
                dout_strm.write(dRight);
                strm_out_end.write(0);
            } else {
                // no operation
            }
            // whether output last left
            if (!output_last_left) {
                kout_strm.write(iLeft);
                dout_strm.write(dLeft);
                strm_out_end.write(0);
            } else {
                // no operation
            }
        } else {
            // whether output last left
            if (!output_last_left) {
                kout_strm.write(iLeft);
                dout_strm.write(dLeft);
                strm_out_end.write(0);
            } else {
                // no operation
            }
            // whether output last right
            if (!output_last_right) {
                kout_strm.write(iRight);
                dout_strm.write(dRight);
                strm_out_end.write(0);
            } else {
                // no operation
            }
        }
    }
    //-------------------------------------end---------------------------------//

    strm_out_end.write(1);
}

template <typename Key_Type>
void merge_sort_top(hls::stream<Key_Type>& left_kin_strm,
                    hls::stream<bool>& left_strm_in_end,

                    hls::stream<Key_Type>& right_kin_strm,
                    hls::stream<bool>& right_strm_in_end,

                    hls::stream<Key_Type>& kout_strm,
                    hls::stream<bool>& strm_out_end,

                    bool sign) {
    Key_Type iLeft = 0;
    Key_Type iRight = 0;
    Key_Type Result = 0;

    bool left_strm_end = 0;
    bool right_strm_end = 0;

    // control output the last element
    bool output_last_left = 0;
    bool output_last_right = 0;

    // read 1st left key
    left_strm_end = left_strm_in_end.read();
    if (!left_strm_end) {
        iLeft = left_kin_strm.read();
        left_strm_end = left_strm_in_end.read();
    } else {
        output_last_left = 1;
    }

    // read 1st right key
    right_strm_end = right_strm_in_end.read();
    if (!right_strm_end) {
        iRight = right_kin_strm.read();
        right_strm_end = right_strm_in_end.read();
    } else {
        output_last_right = 1;
    }

merge_loop:
    while (!left_strm_end || !right_strm_end) {
#pragma HLS PIPELINE II = 1
#pragma HLS loop_tripcount max = 20 min = 20
        // output && read next left or right
        if (sign) {
            //----------------------------------ascending-------------------------------//
            if (iLeft < iRight) {
                if (!left_strm_end) {
                    // output left && read next left
                    Result = iLeft;

                    iLeft = left_kin_strm.read();
                    left_strm_end = left_strm_in_end.read();
                } else if (left_strm_end && !output_last_left && !right_strm_end) {
                    // output last left && no read
                    Result = iLeft;
                    output_last_left = 1;
                } else if (left_strm_end && output_last_left && !right_strm_end) {
                    // output right && read risidual right
                    Result = iRight;

                    iRight = right_kin_strm.read();
                    right_strm_end = right_strm_in_end.read();
                } else {
                    // left_strm_end && right_strm_end
                    // no operation, break loop
                }
            } else {
                if (!right_strm_end) {
                    // output right && read next right
                    Result = iRight;

                    iRight = right_kin_strm.read();
                    right_strm_end = right_strm_in_end.read();
                } else if (right_strm_end && !output_last_right && !left_strm_end) {
                    // output last right && no read
                    Result = iRight;
                    output_last_right = 1;
                } else if (right_strm_end && output_last_right && !left_strm_end) {
                    // output left && read risidual left
                    Result = iLeft;

                    iLeft = left_kin_strm.read();
                    left_strm_end = left_strm_in_end.read();
                } else {
                    // left_strm_end && right_strm_end
                    // no operation, break loop
                }
            }
            //-------------------------------------end---------------------------------//
        } else {
            //----------------------------------descending-----------------------------//
            if (iLeft < iRight) {
                if (!right_strm_end) {
                    // output right && read next right
                    Result = iRight;

                    iRight = right_kin_strm.read();
                    right_strm_end = right_strm_in_end.read();
                } else if (right_strm_end && !output_last_right && !left_strm_end) {
                    // output last right && no read
                    Result = iRight;
                    output_last_right = 1;
                } else if (right_strm_end && output_last_right && !left_strm_end) {
                    // output left && read risidual left
                    Result = iLeft;

                    iLeft = left_kin_strm.read();
                    left_strm_end = left_strm_in_end.read();
                } else {
                    // left_strm_end && right_strm_end
                    // no operation, break loop
                }
            } else {
                if (!left_strm_end) {
                    // output left && read next left
                    Result = iLeft;

                    iLeft = left_kin_strm.read();
                    left_strm_end = left_strm_in_end.read();
                } else if (left_strm_end && !output_last_left && !right_strm_end) {
                    // output last left && no read
                    Result = iLeft;
                    output_last_left = 1;
                } else if (left_strm_end && output_last_left && !right_strm_end) {
                    // output right && read risidual right
                    Result = iRight;

                    iRight = right_kin_strm.read();
                    right_strm_end = right_strm_in_end.read();
                } else {
                    // left_strm_end && right_strm_end
                    // no operation, break loop
                }
            }
            //-------------------------------------end---------------------------------//
        }
        kout_strm.write(Result);
        strm_out_end.write(0);
    }

    //------------------------------output last element------------------------//
    if (sign) {
        if (iLeft < iRight) {
            // whether output last left
            if (!output_last_left) {
                kout_strm.write(iLeft);
                strm_out_end.write(0);
            } else {
                // no operation
            }
            // whether output last right
            if (!output_last_right) {
                kout_strm.write(iRight);
                strm_out_end.write(0);
            } else {
                // no operation
            }
        } else {
            // whether output last right
            if (!output_last_right) {
                kout_strm.write(iRight);
                strm_out_end.write(0);
            } else {
                // no operation
            }
            // whether output last left
            if (!output_last_left) {
                kout_strm.write(iLeft);
                strm_out_end.write(0);
            } else {
                // no operation
            }
        }
    } else {
        if (iLeft < iRight) {
            // whether output last right
            if (!output_last_right) {
                kout_strm.write(iRight);
                strm_out_end.write(0);
            } else {
                // no operation
            }
            // whether output last left
            if (!output_last_left) {
                kout_strm.write(iLeft);
                strm_out_end.write(0);
            } else {
                // no operation
            }
        } else {
            // whether output last left
            if (!output_last_left) {
                kout_strm.write(iLeft);
                strm_out_end.write(0);
            } else {
                // no operation
            }
            // whether output last right
            if (!output_last_right) {
                kout_strm.write(iRight);
                strm_out_end.write(0);
            } else {
                // no operation
            }
        }
    }
    //-------------------------------------end---------------------------------//

    strm_out_end.write(1);
}

} // namespace details
} // namespace database
} // namespace xf

namespace xf {
namespace database {

/**
 * @brief Merge sort function.
 *
 * @tparam Data_Type the input and output key type
 *
 * @param left_kin_strm input key stream
 * @param left_strm_in_end end flag stream for left input
 *
 * @param right_kin_strm input key stream
 * @param right_strm_in_end end flag stream for right input
 *
 * @param kout_strm output key stream
 * @param strm_out_end end flag stream for output data
 *
 * @param order 1:ascending 0:descending
 */
template <typename Key_Type>
void mergeSort(hls::stream<Key_Type>& left_kin_strm,
               hls::stream<bool>& left_strm_in_end,

               hls::stream<Key_Type>& right_kin_strm,
               hls::stream<bool>& right_strm_in_end,

               hls::stream<Key_Type>& kout_strm,
               hls::stream<bool>& strm_out_end,

               bool order) {
    //#pragma HLS INLINE

    details::merge_sort_top<Key_Type>(left_kin_strm, left_strm_in_end, right_kin_strm, right_strm_in_end, kout_strm,
                                      strm_out_end, order);
}

/**
 * @brief Merge sort function.
 *
 * @tparam Data_Type the input and output data type
 * @tparam Data_Type the input and output key type
 *
 * @param left_din_strm input left data stream
 * @param left_kin_strm input key stream
 * @param left_strm_in_end end flag stream for left input
 *
 * @param right_din_strm input right data stream
 * @param right_kin_strm input key stream
 * @param right_strm_in_end end flag stream for right input
 *
 * @param dout_strm output data stream
 * @param kout_strm output key stream
 * @param strm_out_end end flag stream for output data
 *
 * @param order 1:ascending 0:descending
 */
template <typename Data_Type, typename Key_Type>
void mergeSort(hls::stream<Data_Type>& left_din_strm,
               hls::stream<Key_Type>& left_kin_strm,
               hls::stream<bool>& left_strm_in_end,

               hls::stream<Data_Type>& right_din_strm,
               hls::stream<Key_Type>& right_kin_strm,
               hls::stream<bool>& right_strm_in_end,

               hls::stream<Data_Type>& dout_strm,
               hls::stream<Key_Type>& kout_strm,
               hls::stream<bool>& strm_out_end,

               bool order) {
#pragma HLS INLINE

    details::merge_sort_top<Data_Type, Key_Type>(left_din_strm, left_kin_strm, left_strm_in_end, right_din_strm,
                                                 right_kin_strm, right_strm_in_end, dout_strm, kout_strm, strm_out_end,
                                                 order);
}

} // namespace database
} // namespace xf

#endif
