#ifndef __PARAM_H__
#define __PARAM_H__

#define COMMIT_ID_0 0xbd35
#define COMMIT_ID_1 0xbf50
#define CORE_PER_BATCH 32
#define NUM_BATCH 3

#define MAX_BYTES_BUFA 1024
#define MAX_BYTES_BUFB 2048
#define MAX_BYTES_BUFC 1024
#define INTER_OFFSET 1024

#define LAYER_PARAM_BUF_SIZE 16
#define WLUT_BUF_SIZE 64

enum CHIPTYPE{
ES1=0,
ES2=1,
PROD=2};
#define CHIP PROD


#define AIE_ACTIVATIONS_EN 1
//[deprecated]
#define CONV_LEAKYRELU_EN 0

#define PL_FREQ 250
#define VVER 212

#endif /*__PARAM_H__*/
