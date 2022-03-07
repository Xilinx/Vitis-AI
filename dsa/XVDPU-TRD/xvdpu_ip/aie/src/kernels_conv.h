#ifndef __FUNCTION_KERNELS_H__
#define __FUNCTION_KERNELS_H__
#include "param.h"
#if (VVER!=1922 && VVER!=201)
#include <adf.h>
#else
#include <cardano.h>
#endif

void super_kernel_input_casc(
    input_window_int32 * ifms,     // IFMs sub-volume pointer
    input_window_int32 * weights,     // weight sub-volume pointer
	output_stream_acc48* cascadeout);

void super_kernel_casc_output(
    input_window_int32 * ifms, 
    input_window_int32 * weights,
	input_stream_acc48* cascadein,
    output_window_int32 * ofms);

void super_kernel_input_output(input_window_int8 *bufA,				//in 0
							   input_window_int8 *bufB,				//in 1
							   output_window_int8 *bufC);


void super_kernel_casc_casc(input_window_int8 *bufA, 
							input_window_int8 *bufB, 
							input_stream_acc48* cascadein, 
							output_stream_acc48* cascadeout);


#endif
