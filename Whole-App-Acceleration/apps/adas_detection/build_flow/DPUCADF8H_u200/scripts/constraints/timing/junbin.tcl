puts "Start to source [info script]"

global DPU_V3_TOP

##### #set ML_SHELL_TOP pfm_top_i/dynamic_region/dpdpuv3_wrapper_1/inst
##### global ML_SHELL_TOP 
##### 
##### set_multicycle_path 5 -setup -from [get_pins -of [get_cells -hier -filter {PRIMITIVE_TYPE =~ REGISTER.SDR.*}] -filter name=~$DPU_V3_TOP/core_array[*].u_core/u_conv_top_core/u_conv_pad_core/pe_bias_rr_reg[*]/C] -to [get_pins -of [get_cells -hier -filter {PRIMITIVE_TYPE =~ REGISTER.SDR.*}] -filter name=~$DPU_V3_TOP/core_array[*].u_core/u_conv_top_core/u_addbias/addbias[*].res_reg[*]/D]
##### set_multicycle_path 4 -hold  -from [get_pins -of [get_cells -hier -filter {PRIMITIVE_TYPE =~ REGISTER.SDR.*}] -filter name=~$DPU_V3_TOP/core_array[*].u_core/u_conv_top_core/u_conv_pad_core/pe_bias_rr_reg[*]/C] -to [get_pins -of [get_cells -hier -filter {PRIMITIVE_TYPE =~ REGISTER.SDR.*}] -filter name=~$DPU_V3_TOP/core_array[*].u_core/u_conv_top_core/u_addbias/addbias[*].res_reg[*]/D]

#set ML_SHELL_TOP pfm_top_i/dynamic_region/dpdpuv3_wrapper_1/inst

#global ML_SHELL_TOP
set_multicycle_path 4 -setup -start -from [get_pins -of [get_cells -hier -filter {PRIMITIVE_TYPE =~ REGISTER.SDR.*}] -filter name=~$DPU_V3_TOP/core_array[*].u_core/u_conv_top_core/u_conv_pad_core/cho[*].pe_bias_r_reg[*]/C] -to [get_pins -of [get_cells -hier -filter {PRIMITIVE_TYPE =~ REGISTER.SDR.*}] -filter name=~$DPU_V3_TOP/core_array[*].u_core/u_conv_top_core/u_conv_pad_core/pe_bias_rr_reg[*]/D]
set_multicycle_path 3 -hold -from [get_pins -of [get_cells -hier -filter {PRIMITIVE_TYPE =~ REGISTER.SDR.*}] -filter name=~$DPU_V3_TOP/core_array[*].u_core/u_conv_top_core/u_conv_pad_core/cho[*].pe_bias_r_reg[*]/C] -to [get_pins -of [get_cells -hier -filter {PRIMITIVE_TYPE =~ REGISTER.SDR.*}] -filter name=~$DPU_V3_TOP/core_array[*].u_core/u_conv_top_core/u_conv_pad_core/pe_bias_rr_reg[*]/D]
#set_multicycle_path 4 -setup -start -from [get_pins -of [get_cells -hier -filter {PRIMITIVE_TYPE =~ REGISTER.SDR.*}] -filter name=~$DPU_V3_TOP/core_array[*].u_core/u_conv_top_core/u_conv_pad_core/cho[*].pe_prelu_r_reg[*]/C] -to [get_pins -of [get_cells -hier -filter {PRIMITIVE_TYPE =~ REGISTER.SDR.*}] -filter name=~$DPU_V3_TOP/core_array[*].u_core/u_conv_top_core/u_conv_pad_core/pe_prelu_rr_reg[*]/D]
#set_multicycle_path 3 -hold -from [get_pins -of [get_cells -hier -filter {PRIMITIVE_TYPE =~ REGISTER.SDR.*}] -filter name=~$DPU_V3_TOP/core_array[*].u_core/u_conv_top_core/u_conv_pad_core/cho[*].pe_prelu_r_reg[*]/C] -to [get_pins -of [get_cells -hier -filter {PRIMITIVE_TYPE =~ REGISTER.SDR.*}] -filter name=~$DPU_V3_TOP/core_array[*].u_core/u_conv_top_core/u_conv_pad_core/pe_prelu_rr_reg[*]/D]

# set_multicycle_path 4 -setup -start -from [get_pins -of [get_cells -hier -filter {PRIMITIVE_TYPE =~ REGISTER.SDR.*}] -filter name=~$DPU_V3_TOP/core_array[*].u_core/u_conv_top_core/u_fetch_res/data0_reg[*]/C] -to [get_pins -of [get_cells -hier -filter {PRIMITIVE_TYPE =~ REGISTER.SDR.*}] -filter name=~$DPU_V3_TOP/core_array[*].u_core/u_conv_top_core/u_fetch_res/data_d_reg[*]/D]
# set_multicycle_path 3 -hold -from [get_pins -of [get_cells -hier -filter {PRIMITIVE_TYPE =~ REGISTER.SDR.*}] -filter name=~$DPU_V3_TOP/core_array[*].u_core/u_conv_top_core/u_fetch_res/data0_reg[*]/C] -to [get_pins -of [get_cells -hier -filter {PRIMITIVE_TYPE =~ REGISTER.SDR.*}] -filter name=~$DPU_V3_TOP/core_array[*].u_core/u_conv_top_core/u_fetch_res/data_d_reg[*]/D]
# 
# set_multicycle_path 4 -setup -start -from [get_pins -of [get_cells -hier -filter {PRIMITIVE_TYPE =~ REGISTER.SDR.*}] -filter name=~$DPU_V3_TOP/core_array[*].u_core/u_conv_top_core/u_fetch_res/data1_reg[*]/C] -to [get_pins -of [get_cells -hier -filter {PRIMITIVE_TYPE =~ REGISTER.SDR.*}] -filter name=~$DPU_V3_TOP/core_array[*].u_core/u_conv_top_core/u_fetch_res/data_d_reg[*]/D]
# set_multicycle_path 3 -hold -from [get_pins -of [get_cells -hier -filter {PRIMITIVE_TYPE =~ REGISTER.SDR.*}] -filter name=~$DPU_V3_TOP/core_array[*].u_core/u_conv_top_core/u_fetch_res/data1_reg[*]/C] -to [get_pins -of [get_cells -hier -filter {PRIMITIVE_TYPE =~ REGISTER.SDR.*}] -filter name=~$DPU_V3_TOP/core_array[*].u_core/u_conv_top_core/u_fetch_res/data_d_reg[*]/D]


#10-15
#for {set i 10} {$i<=16} {incr i} {
# if {$i==16||$i>=10&&$i<=14} {
# set_multicycle_path 3 -setup -start -from [get_pins -of [get_cells -hier -filter {PRIMITIVE_TYPE =~ REGISTER.SDR.*}] -filter name=~$DPU_V3_TOP/u_conv_com/u_conv_parser/ins_data_convinit_reg[$i]/C] -to [get_pins -of [get_cells -hier -filter {PRIMITIVE_TYPE =~ REGISTER.SDR.*}] -filter name=~$DPU_V3_TOP/core_array[*].u_core/u_conv_top_core/nonl_win[*].nonl_cho[*].u_non_linear/nl_relu_reg[*]/D]
# set_multicycle_path 2 -hold -from [get_pins -of [get_cells -hier -filter {PRIMITIVE_TYPE =~ REGISTER.SDR.*}] -filter name=~$DPU_V3_TOP/u_conv_com/u_conv_parser/ins_data_convinit_reg[$i]/C] -to [get_pins -of [get_cells -hier -filter {PRIMITIVE_TYPE =~ REGISTER.SDR.*}] -filter name=~$DPU_V3_TOP/core_array[*].u_core/u_conv_top_core/nonl_win[*].nonl_cho[*].u_non_linear/nl_relu_reg[*]/D]
# }
#}








set_multicycle_path 4 -setup -start -from [get_pins -of [get_cells -hier -filter {PRIMITIVE_TYPE =~ REGISTER.SDR.*}] -filter name=~$DPU_V3_TOP/core_array[*].u_core/u_conv_top_core/u_fetch_res/data0_reg[*]/C] -through [get_pins -of [get_cells -hier -filter {PRIMITIVE_TYPE =~ REGISTER.SDR.*}] -filter name=~$DPU_V3_TOP/core_array[*].u_core/u_conv_top_core/u_fetch_res/data0_reg[*]/Q]
set_multicycle_path 3 -hold -from [get_pins -of [get_cells -hier -filter {PRIMITIVE_TYPE =~ REGISTER.SDR.*}] -filter name=~$DPU_V3_TOP/core_array[*].u_core/u_conv_top_core/u_fetch_res/data0_reg[*]/C] -through [get_pins -of [get_cells -hier -filter {PRIMITIVE_TYPE =~ REGISTER.SDR.*}] -filter name=~$DPU_V3_TOP/core_array[*].u_core/u_conv_top_core/u_fetch_res/data0_reg[*]/Q]

set_multicycle_path 4 -setup -start -from [get_pins -of [get_cells -hier -filter {PRIMITIVE_TYPE =~ REGISTER.SDR.*}] -filter name=~$DPU_V3_TOP/core_array[*].u_core/u_conv_top_core/u_fetch_res/data1_reg[*]/C] -through [get_pins -of [get_cells -hier -filter {PRIMITIVE_TYPE =~ REGISTER.SDR.*}] -filter name=~$DPU_V3_TOP/core_array[*].u_core/u_conv_top_core/u_fetch_res/data1_reg[*]/Q]
set_multicycle_path 3 -hold -from [get_pins -of [get_cells -hier -filter {PRIMITIVE_TYPE =~ REGISTER.SDR.*}] -filter name=~$DPU_V3_TOP/core_array[*].u_core/u_conv_top_core/u_fetch_res/data1_reg[*]/C] -through [get_pins -of [get_cells -hier -filter {PRIMITIVE_TYPE =~ REGISTER.SDR.*}] -filter name=~$DPU_V3_TOP/core_array[*].u_core/u_conv_top_core/u_fetch_res/data1_reg[*]/Q]
