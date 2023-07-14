//Setting the arch of DPU, For more details, Please read the PG338 


/*====== Architecture Options ======*/
// |------------------------------------------------------|
// | Support 8 DPU size
// | It relates to model. if change, must update model
// +------------------------------------------------------+
// | `define B512               
// +------------------------------------------------------+
// | `define B800                 
// +------------------------------------------------------+
// | `define B1024                 
// +------------------------------------------------------+
// | `define B1152                 
// +------------------------------------------------------+
// | `define B1600                 
// +------------------------------------------------------+
// | `define B2304                 
// +------------------------------------------------------+
// | `define B3136                 
// +------------------------------------------------------+
// | `define B4096                 
// |------------------------------------------------------|

`define B4096 

// |------------------------------------------------------|
// | If the FPGA has Uram. You can define URAM_EN parameter               
// | if change, Don't need update model
// +------------------------------------------------------+
// | for zcu104 : `define URAM_ENABLE               
// +------------------------------------------------------+
// | for zcu102 : `define URAM_DISABLE                 
// |------------------------------------------------------|

`define URAM_DISABLE 

//config URAM
`ifdef URAM_ENABLE
    `define def_UBANK_IMG_N          5
    `define def_UBANK_WGT_N          17
    `define def_UBANK_BIAS           1
`elsif URAM_DISABLE
    `define def_UBANK_IMG_N          0
    `define def_UBANK_WGT_N          0
    `define def_UBANK_BIAS           0
`endif

// |------------------------------------------------------|
// | You can use DRAM if FPGA has extra LUTs               
// | if change, Don't need update model
// +------------------------------------------------------+
// | Enable DRAM  : `define DRAM_ENABLE               
// +------------------------------------------------------+
// | Disable DRAM : `define DRAM_DISABLE                 
// |------------------------------------------------------|

`define DRAM_DISABLE 

//config DRAM
`ifdef DRAM_ENABLE
    `define def_DBANK_IMG_N          1 
    `define def_DBANK_WGT_N          1
    `define def_DBANK_BIAS           1
`elsif DRAM_DISABLE
    `define def_DBANK_IMG_N          0
    `define def_DBANK_WGT_N          0
    `define def_DBANK_BIAS           0
`endif

// |------------------------------------------------------|
// | RAM Usage Configuration              
// | It relates to model. if change, must update model
// +------------------------------------------------------+
// | RAM Usage High : `define RAM_USAGE_HIGH               
// +------------------------------------------------------+
// | RAM Usage Low  : `define RAM_USAGE_LOW                 
// |------------------------------------------------------|

`define RAM_USAGE_LOW

// |------------------------------------------------------|
// | Channel Augmentation Configuration
// | It relates to model. if change, must update model
// +------------------------------------------------------+
// | Enable  : `define CHANNEL_AUGMENTATION_ENABLE              
// +------------------------------------------------------+
// | Disable : `define CHANNEL_AUGMENTATION_DISABLE                
// |------------------------------------------------------|

`define CHANNEL_AUGMENTATION_ENABLE

// |------------------------------------------------------|
// | ALU parallel Configuration
// | It relates to model. if change, must update model
// +------------------------------------------------------+
// | setting 0  : `define ALU_PARALLEL_DEFAULT              
// +------------------------------------------------------+
// | setting 1  : `define ALU_PARALLEL_1                
// |------------------------------------------------------|
// | setting 2  : `define ALU_PARALLEL_2                
// |------------------------------------------------------|
// | setting 3  : `define ALU_PARALLEL_4                
// |------------------------------------------------------|
// | setting 4  : `define ALU_PARALLEL_8                
// |------------------------------------------------------|

`define ALU_PARALLEL_DEFAULT 

// +------------------------------------------------------+
// | CONV RELU Type Configuration
// | It relates to model. if change, must update model
// +------------------------------------------------------+
// | `define CONV_RELU_RELU6
// +------------------------------------------------------+
// | `define CONV_RELU_LEAKYRELU_RELU6
// |------------------------------------------------------|

`define CONV_RELU_LEAKYRELU_RELU6

// +------------------------------------------------------+
// | ALU RELU Type Configuration
// | It relates to model. if change, must update model
// +------------------------------------------------------+
// | `define ALU_RELU_RELU6
// +------------------------------------------------------+
// | `define ALU_RELU_LEAKYRELU_RELU6
// |------------------------------------------------------|

`define ALU_RELU_RELU6

// |------------------------------------------------------|
// | DSP48 Usage Configuration  
// | Use dsp replace of lut in conv operate 
// | if change, Don't need update model
// +------------------------------------------------------+
// | `define DSP48_USAGE_HIGH              
// +------------------------------------------------------+
// | `define DSP48_USAGE_LOW                
// |------------------------------------------------------|

`define DSP48_USAGE_HIGH 

// |------------------------------------------------------|
// | Power Configuration
// | if change, Don't need update model
// +------------------------------------------------------+
// | `define LOWPOWER_ENABLE              
// +------------------------------------------------------+
// | `define LOWPOWER_DISABLE               
// |------------------------------------------------------|

`define LOWPOWER_DISABLE

// |------------------------------------------------------|
// | DEVICE Configuration
// | if change, Don't need update model
// +------------------------------------------------------+
// | `define MPSOC              
// +------------------------------------------------------+
// | `define ZYNQ7000               
// |------------------------------------------------------|

`define MPSOC
  



 
