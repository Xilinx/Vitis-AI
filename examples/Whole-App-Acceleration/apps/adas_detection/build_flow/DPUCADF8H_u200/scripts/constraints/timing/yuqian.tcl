puts "Start to source [info script]"

#set DPU_V3_TOP        pfm_top_i/dynamic_region/dpdpuv3_wrapper_1/inst/u_dpdpuv3_top

# false path
set_false_path -from    [get_pins -of [get_cells -hier -filter {PRIMITIVE_TYPE =~ REGISTER.SDR.*}] -filter name=~$DPU_V3_TOP/u_flowctrl/reg_*/C]    -to     [get_pins -of [get_cells -hier -filter {PRIMITIVE_TYPE =~ REGISTER.SDR.*}] -filter name=~$DPU_V3_TOP/u_scheduler/u_profiler/*/R]

set_false_path -from    [get_pins -of [get_cells -hier -filter {PRIMITIVE_TYPE =~ REGISTER.SDR.*}] -filter name=~$DPU_V3_TOP/u_flowctrl/reg_*/C]    -to     [get_pins -of [get_cells -hier -filter {PRIMITIVE_TYPE =~ REGISTER.SDR.*}] -filter name=~$DPU_V3_TOP/u_scheduler/u_profiler/*/D]


set_false_path -from    [get_pins -of [get_cells -hier -filter {PRIMITIVE_TYPE =~ REGISTER.SDR.*}] -filter name=~$DPU_V3_TOP/u_flowctrl/reg_*/C]    -to     [get_pins -of [get_cells -hier -filter {PRIMITIVE_TYPE =~ REGISTER.SDR.*}] -filter name=~$DPU_V3_TOP/u_scheduler/u_profiler/*/CE]


set_false_path -from    [get_pins -of [get_cells -hier -filter {PRIMITIVE_TYPE =~ REGISTER.SDR.*}] -filter name=~$DPU_V3_TOP/u_flowctrl/reg_*/C]    -to     [get_pins -of [get_cells -hier -filter {PRIMITIVE_TYPE =~ REGISTER.SDR.*}] -filter name=~$DPU_V3_TOP/u_scheduler/u_fetch/*/D]


set_false_path -to      [get_pins -of [get_cells -hier -filter {PRIMITIVE_TYPE =~ REGISTER.SDR.*}] -filter name=~$DPU_V3_TOP/u_flowctrl/rdata*/D]   -from   [get_pins -of [get_cells -hier -filter {PRIMITIVE_TYPE =~ REGISTER.SDR.*}] -filter name=~$DPU_V3_TOP/u_scheduler/u_profiler/*/C]


set_false_path -from    [get_pins -of [get_cells -hier -filter {PRIMITIVE_TYPE =~ REGISTER.SDR.*}] -filter name=~$DPU_V3_TOP/u_flowctrl/reg_awcache_reg*/C]
set_false_path -from    [get_pins -of [get_cells -hier -filter {PRIMITIVE_TYPE =~ REGISTER.SDR.*}] -filter name=~$DPU_V3_TOP/u_flowctrl/reg_arcache_reg*/C]

#mcp 
#profiler
set_multicycle_path 2 -setup -from [get_pins -of [get_cells -hier -filter {PRIMITIVE_TYPE =~ REGISTER.SDR.*}] -filter name=~$DPU_V3_TOP/u_flowctrl/task_*/C]    -to     [get_pins -of [get_cells -hier -filter {PRIMITIVE_TYPE =~ REGISTER.SDR.*}] -filter name=~$DPU_V3_TOP/u_scheduler/u_profiler/*/D]
set_multicycle_path 1 -hold -end -from [get_pins -of [get_cells -hier -filter {PRIMITIVE_TYPE =~ REGISTER.SDR.*}] -filter name=~$DPU_V3_TOP/u_flowctrl/task_*/C]    -to     [get_pins -of [get_cells -hier -filter {PRIMITIVE_TYPE =~ REGISTER.SDR.*}] -filter name=~$DPU_V3_TOP/u_scheduler/u_profiler/*/D]

set_multicycle_path 2 -setup -from [get_pins -of [get_cells -hier -filter {PRIMITIVE_TYPE =~ REGISTER.SDR.*}] -filter name=~$DPU_V3_TOP/u_flowctrl/task_*/C]    -to     [get_pins -of [get_cells -hier -filter {PRIMITIVE_TYPE =~ REGISTER.SDR.*}] -filter name=~$DPU_V3_TOP/u_scheduler/u_fetch/*/D]
set_multicycle_path 1 -hold -end -from [get_pins -of [get_cells -hier -filter {PRIMITIVE_TYPE =~ REGISTER.SDR.*}] -filter name=~$DPU_V3_TOP/u_flowctrl/task_*/C]    -to     [get_pins -of [get_cells -hier -filter {PRIMITIVE_TYPE =~ REGISTER.SDR.*}] -filter name=~$DPU_V3_TOP/u_scheduler/u_fetch/*/D]



#tmp
#
set_false_path -to      [get_pins -of [get_cells -hier -filter {PRIMITIVE_TYPE =~ REGISTER.SDR.*}] -filter name=~$DPU_V3_TOP/u_save/u_ddr_writer/axvalid_reg/D]
set_false_path -to      [get_pins -of [get_cells -hier -filter {PRIMITIVE_TYPE =~ REGISTER.SDR.*}] -filter name=~$DPU_V3_TOP/u_save/u_ddr_writer/axaddr_reg[*]/D]
set_false_path -to      [get_pins -of [get_cells -hier -filter {PRIMITIVE_TYPE =~ REGISTER.SDR.*}] -filter name=~$DPU_V3_TOP/u_load/u_ddr_reader/axvalid_reg/D]
set_false_path -to      [get_pins -of [get_cells -hier -filter {PRIMITIVE_TYPE =~ REGISTER.SDR.*}] -filter name=~$DPU_V3_TOP/u_load/u_ddr_reader/axaddr_reg[*]/D]
set_false_path -to      [get_pins -of [get_cells -hier -filter {PRIMITIVE_TYPE =~ REGISTER.SDR.*}] -filter name=~$DPU_V3_TOP/u_save/u_ddr_writer/axlen_reg[*]/D]
set_false_path -to      [get_pins -of [get_cells -hier -filter {PRIMITIVE_TYPE =~ REGISTER.SDR.*}] -filter name=~$DPU_V3_TOP/u_save/u_ddr_writer/axlen_reg[*]/CE]
set_false_path -to      [get_pins -of [get_cells -hier -filter {PRIMITIVE_TYPE =~ REGISTER.SDR.*}] -filter name=~$DPU_V3_TOP/u_load/u_ddr_reader/axlen_reg[*]/D]
set_false_path -to      [get_pins -of [get_cells -hier -filter {PRIMITIVE_TYPE =~ REGISTER.SDR.*}] -filter name=~$DPU_V3_TOP/u_load/u_ddr_reader/axlen_reg[*]/CE]
#
#tmp


