
`ifndef ISA_1_7_0
    `define ISA_1_7_0 TRUE
`endif

`ifndef POOL_KRNL_5X5_DISABLE
    `define POOL_KRNL_5X5_SUPPORT TRUE
`endif
`ifndef POOL_KRNL_8X8_DISABLE
    `define POOL_KRNL_8X8_SUPPORT TRUE
`endif

`ifndef MISC_PARAMETERS 
    `define MISC_PARAMETERS \
            .DATA_W             ( DATA_W             ), \
            .CH_N               ( CH_N               ), \
            .CONV_PP_N          ( CONV_PP_N          ), \
            .MISC_PP_N          ( MISC_PP_N          ), \
            .IMG_BID_W          ( IMG_BID_W          ), \
            .IMG_ADDR_W         ( IMG_ADDR_W         ), \
            .JUMP_M1_W          ( JUMP_M1_W          ), \
            .KERNEL_M1_W        ( KERNEL_M1_W        ), \
            .SHIFT_CUT_W        ( SHIFT_CUT_W        ), \
            .POOL_TYPE_W        ( POOL_TYPE_W        ), \
            .STRIDE_M1_W        ( STRIDE_M1_W        ), \
            .STRIDE_OFFSET_W    ( STRIDE_OFFSET_W    ), \
            .VPP_W              ( VPP_W              ), \
            .STRIDE_OUT_M1_W    ( STRIDE_OUT_M1_W    ), \
            .PAD_W              ( PAD_W              ), \
            .CHANNEL_GROUP_M1_W ( CHANNEL_GROUP_M1_W ), \
            .JUMP_ENDL_M1_W     ( JUMP_ENDL_M1_W     ), \
            .BANK_ID_W          ( BANK_ID_W          ), \
            .LENGTH_M1_W        ( LENGTH_M1_W        ), \
            .BANK_ADDR_W        ( BANK_ADDR_W        ), \
            .NUM_FEATURE_MAP    ( NUM_FEATURE_MAP    ), \
            .SHIFT_READ_W       ( SHIFT_READ_W       ), \
            .ID_W               ( ID_W               ), \
            .SHIFT_WRITE_W      ( SHIFT_WRITE_W      ), \
            .ACT_TYPE_W         ( ACT_TYPE_W         ), \
            .NUM_M1_W           ( NUM_M1_W           ), \
            .EXT_W              ( EXT_W              ), \
            .PRECISION_W        ( PRECISION_W        ), \
            .LB_FIFO_S0         ( LB_FIFO_S0         ), \
            .LB_FIFO_S1         ( LB_FIFO_S1         ), \
            .LB_FIFO_D          ( LB_FIFO_D          )
`endif

`ifdef ISA_1_7_0
    `ifndef MISC_PARAMETERS_STATEMENT 
        `define MISC_PARAMETERS_STATEMENT \
            parameter DATA_W             = 8  , \
            parameter CH_N               = 32 , \
            parameter CONV_PP_N          = 4  , \
            parameter MISC_PP_N          = 2  , \
            parameter IMG_BID_W          = 4  , \
            parameter IMG_ADDR_W         = 23 , \
            parameter JUMP_M1_W          = 10 , \
            parameter KERNEL_M1_W        = 3  , \
            parameter SHIFT_CUT_W        = 4  , \
            parameter POOL_TYPE_W        = 2  , \
            parameter STRIDE_M1_W        = 3  , \
            parameter STRIDE_OFFSET_W    = 3  , \
            parameter VPP_W              = 3  , \
            parameter STRIDE_OUT_M1_W    = 4  , \
            parameter PAD_W              = 3  , \
            parameter CHANNEL_GROUP_M1_W = 8  , \
            parameter JUMP_ENDL_M1_W     = 16 , \
            parameter BANK_ID_W          = 6  , \
            parameter LENGTH_M1_W        = 12 , \
            parameter BANK_ADDR_W        = 24 , \
            parameter NUM_FEATURE_MAP    = 4  , \
            parameter SHIFT_READ_W       = 4  , \
            parameter ID_W               = 2  , \
            parameter SHIFT_WRITE_W      = 4  , \
            parameter ACT_TYPE_W         = 1  , \
            parameter NUM_M1_W           = 2  , \
            parameter EXT_W              = KERNEL_M1_W + KERNEL_M1_W , \
            parameter EXT_CI_W           = 0  , \
            parameter EXT_CO_W           = 0  , \
            parameter PRECISION_W        = 2  , \
            parameter LB_FIFO_S0         = 1  , \
            parameter LB_FIFO_S1         = 1  , \
            parameter LB_FIFO_D          = 512 , \
            parameter DIST_RAM_S         = 0 , \
            parameter DIST_RAM_D         = 32
    `endif
`endif 

`ifdef ISA_1_4_0_MULTI_ELEW
    `ifndef MISC_PARAMETERS_STATEMENT 
        `define MISC_PARAMETERS_STATEMENT \
            parameter DATA_W             = 8  , \
            parameter CH_N               = 16 , \
            parameter CONV_PP_N          = 8  , \
            parameter MISC_PP_N          = 1  , \
            parameter IMG_BID_W          = 4  , \
            parameter IMG_ADDR_W         = 13 , \
            parameter JUMP_M1_W          = 10 , \
            parameter KERNEL_M1_W        = 3  , \
            parameter SHIFT_CUT_W        = 4  , \
            parameter POOL_TYPE_W        = 2  , \
            parameter STRIDE_M1_W        = 3  , \
            parameter STRIDE_OFFSET_W    = 3  , \
            parameter VPP_W              = 3  , \
            parameter STRIDE_OUT_M1_W    = 4  , \
            parameter PAD_W              = 3  , \
            parameter CHANNEL_GROUP_M1_W = 8  , \
            parameter JUMP_ENDL_M1_W     = 13 , \
            parameter BANK_ID_W          = 6  , \
            parameter LENGTH_M1_W        = 10 , \
            parameter BANK_ADDR_W        = 13 , \
            parameter NUM_FEATURE_MAP    = 4  , \
            parameter SHIFT_READ_W       = 4  , \
            parameter ID_W               = 2  , \
            parameter SHIFT_WRITE_W      = 4  , \
            parameter ACT_TYPE_W         = 1  , \
            parameter NUM_M1_W           = 2  , \
            parameter EXT_W              = KERNEL_M1_W + KERNEL_M1_W , \
            parameter EXT_CI_W           = 0  , \
            parameter EXT_CO_W           = 0  , \
            parameter PRECISION_W        = 2  , \
            parameter LB_FIFO_S0         = 1  , \
            parameter LB_FIFO_S1         = 1  , \
            parameter LB_FIFO_D          = 512 , \
            parameter DIST_RAM_S         = 0 , \
            parameter DIST_RAM_D         = 32
    `endif
`endif 

