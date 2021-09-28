
#ifndef XF_MACHINE_DUT_H
#define XF_MACHINE_DUT_H

#include "ap_int.h"

extern "C" void naiveBayesTrain_kernel(const int num_of_class,
                                       const int num_of_terms,
                                       ap_uint<512>* buf_in,
                                       ap_uint<512>* buf_out0,
                                       ap_uint<512>* buf_out1);

#endif
