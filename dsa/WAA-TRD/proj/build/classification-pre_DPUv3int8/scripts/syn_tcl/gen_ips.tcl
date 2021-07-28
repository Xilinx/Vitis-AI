puts "Start to source [info script]"


#####################################
# clk_wiz
# 
#create_ip -name clk_wiz -vendor xilinx.com -library ip -version 6.0 -module_name clk_wiz
create_ip -name clk_wiz -vendor xilinx.com -library ip -module_name clk_wiz
if { $SHELL_VER =="aws" } {
    if { $FREQ==600 } {
        set_property -dict [list CONFIG.Component_Name {clk_wiz} CONFIG.PRIM_IN_FREQ {300.000} CONFIG.PRIMARY_PORT {clk_in_300} CONFIG.CLK_OUT1_PORT {clk_out} CONFIG.CLKOUT1_REQUESTED_OUT_FREQ {600.000} CONFIG.RESET_TYPE {ACTIVE_LOW} CONFIG.MMCM_CLKIN1_PERIOD {3.333} CONFIG.MMCM_CLKOUT0_DIVIDE_F {2.000} CONFIG.RESET_PORT {resetn}] [get_ips clk_wiz]
        set_property -dict [list CONFIG.PRIM_IN_FREQ {250.000} CONFIG.CLKIN1_JITTER_PS {40.0} CONFIG.MMCM_DIVCLK_DIVIDE {5} CONFIG.MMCM_CLKFBOUT_MULT_F {24.000} CONFIG.MMCM_CLKIN1_PERIOD {4.000} CONFIG.CLKOUT1_JITTER {99.082} CONFIG.CLKOUT1_PHASE_ERROR {154.678}] [get_ips clk_wiz]
    } elseif { $FREQ==500 } {
        set_property -dict [list CONFIG.Component_Name {clk_wiz} CONFIG.PRIM_IN_FREQ {300.000} CONFIG.PRIMARY_PORT {clk_in_300} CONFIG.CLK_OUT1_PORT {clk_out} CONFIG.CLKOUT1_REQUESTED_OUT_FREQ {600.000} CONFIG.RESET_TYPE {ACTIVE_LOW} CONFIG.MMCM_CLKIN1_PERIOD {3.333} CONFIG.MMCM_CLKOUT0_DIVIDE_F {2.000} CONFIG.RESET_PORT {resetn}] [get_ips clk_wiz]
        set_property -dict [list CONFIG.PRIM_IN_FREQ {250.000} CONFIG.CLKOUT1_REQUESTED_OUT_FREQ {500.000} CONFIG.CLKIN1_JITTER_PS {40.0} CONFIG.MMCM_DIVCLK_DIVIDE {1} CONFIG.MMCM_CLKFBOUT_MULT_F {4.750} CONFIG.MMCM_CLKIN1_PERIOD {4.000} CONFIG.MMCM_CLKOUT0_DIVIDE_F {2.375} CONFIG.CLKOUT1_JITTER {74.376} CONFIG.CLKOUT1_PHASE_ERROR {78.266}] [get_ips clk_wiz]
    }
} elseif { $FREQ==600 } {
set_property -dict [list CONFIG.Component_Name {clk_wiz} CONFIG.PRIM_IN_FREQ {300.000} CONFIG.PRIMARY_PORT {clk_in_300} CONFIG.CLK_OUT1_PORT {clk_out} CONFIG.CLKOUT1_REQUESTED_OUT_FREQ {600.000} CONFIG.RESET_TYPE {ACTIVE_LOW} CONFIG.MMCM_CLKIN1_PERIOD {3.333} CONFIG.MMCM_CLKOUT0_DIVIDE_F {2.000} CONFIG.RESET_PORT {resetn}] [get_ips clk_wiz]
} elseif { $FREQ==500 } {
set_property -dict [list CONFIG.Component_Name {clk_wiz} CONFIG.PRIM_IN_FREQ {300.000} CONFIG.PRIMARY_PORT {clk_in_300} CONFIG.CLK_OUT1_PORT {clk_out} CONFIG.CLKOUT1_REQUESTED_OUT_FREQ {500.000} CONFIG.CLKIN1_JITTER_PS {33.330000000000005} CONFIG.MMCM_DIVCLK_DIVIDE {3} CONFIG.MMCM_CLKFBOUT_MULT_F {11.875} CONFIG.MMCM_CLKIN1_PERIOD {3.333} CONFIG.MMCM_CLKIN2_PERIOD {10.0} CONFIG.MMCM_CLKOUT0_DIVIDE_F {2.375} CONFIG.CLKOUT1_JITTER {85.420} CONFIG.CLKOUT1_PHASE_ERROR {87.466}] [get_ips clk_wiz]
} elseif { $FREQ==400 } {
set_property -dict [list CONFIG.Component_Name {clk_wiz} CONFIG.PRIM_IN_FREQ {300.000} CONFIG.PRIMARY_PORT {clk_in_300} CONFIG.CLK_OUT1_PORT {clk_out} CONFIG.CLKOUT1_REQUESTED_OUT_FREQ {400.000} CONFIG.RESET_TYPE {ACTIVE_LOW} CONFIG.CLKIN1_JITTER_PS {33.330000000000005} CONFIG.MMCM_CLKFBOUT_MULT_F {4.000} CONFIG.MMCM_CLKIN1_PERIOD {3.333} CONFIG.MMCM_CLKIN2_PERIOD {10.0} CONFIG.MMCM_CLKOUT0_DIVIDE_F {3.000} CONFIG.RESET_PORT {resetn} CONFIG.CLKOUT1_JITTER {77.334} CONFIG.CLKOUT1_PHASE_ERROR {77.836}] [get_ips clk_wiz]
} elseif { $FREQ==50 } {
set_property -dict [list CONFIG.Component_Name {clk_wiz} CONFIG.PRIM_IN_FREQ {300.000} CONFIG.PRIMARY_PORT {clk_in_300} CONFIG.CLK_OUT1_PORT {clk_out} CONFIG.CLKOUT1_REQUESTED_OUT_FREQ {50.000} CONFIG.RESET_TYPE {ACTIVE_LOW} CONFIG.CLKIN1_JITTER_PS {33.330000000000005} CONFIG.MMCM_CLKFBOUT_MULT_F {4.000} CONFIG.MMCM_CLKIN1_PERIOD {3.333} CONFIG.MMCM_CLKIN2_PERIOD {10.0} CONFIG.MMCM_CLKOUT0_DIVIDE_F {24.000} CONFIG.RESET_PORT {resetn} CONFIG.CLKOUT1_JITTER {116.415} CONFIG.CLKOUT1_PHASE_ERROR {77.836}] [get_ips clk_wiz]
}

#####################################
#
# axi_lite_clock_converter_32
#create_ip -name axi_clock_converter -vendor xilinx.com -library ip -version 2.1 -module_name axi_lite_clock_converter_32
create_ip -name axi_clock_converter -vendor xilinx.com -library ip -module_name axi_lite_clock_converter_32
set_property -dict [list CONFIG.Component_Name {axi_lite_clock_converter_32} CONFIG.PROTOCOL {AXI4LITE} CONFIG.ADDR_WIDTH {64} CONFIG.DATA_WIDTH {32} CONFIG.ID_WIDTH {0} CONFIG.AWUSER_WIDTH {0} CONFIG.ARUSER_WIDTH {0} CONFIG.RUSER_WIDTH {0} CONFIG.WUSER_WIDTH {0} CONFIG.BUSER_WIDTH {0}] [get_ips axi_lite_clock_converter_32]


#####################################
# axi_clock_converter_512
#create_ip -name axi_clock_converter -vendor xilinx.com -library ip -version 2.1 -module_name axi_clock_converter_51 2
create_ip -name axi_clock_converter -vendor xilinx.com -library ip -module_name axi_clock_converter_512
set_property -dict [list CONFIG.Component_Name {axi_clock_converter_512} CONFIG.ADDR_WIDTH {64} CONFIG.DATA_WIDTH {512} CONFIG.ID_WIDTH {8}] [get_ips axi_clock_converter_512]


#####################################
# axi_register_slice_32
#create_ip -name axi_register_slice -vendor xilinx.com -library ip -version 2.1 -module_name axi_register_slice_32
create_ip -name axi_register_slice -vendor xilinx.com -library ip  -module_name axi_register_slice_32
set_property -dict [list CONFIG.READ_WRITE_MODE {READ_ONLY} CONFIG.ADDR_WIDTH {64} CONFIG.ID_WIDTH {8} CONFIG.REG_AR {1} CONFIG.MAX_BURST_LENGTH {256} CONFIG.NUM_READ_OUTSTANDING {32} CONFIG.NUM_WRITE_OUTSTANDING {0}] [get_ips axi_register_slice_32]


#####################################
# axi_register_slice_512
#create_ip -name axi_register_slice -vendor xilinx.com -library ip -version 2.1 -module_name axi_register_slice_512
create_ip -name axi_register_slice -vendor xilinx.com -library ip -module_name axi_register_slice_512
set_property -dict [list CONFIG.ADDR_WIDTH {64} CONFIG.DATA_WIDTH {512} CONFIG.ID_WIDTH {8} CONFIG.MAX_BURST_LENGTH {256} CONFIG.NUM_READ_OUTSTANDING {32} CONFIG.NUM_WRITE_OUTSTANDING {32} CONFIG.Component_Name {axi_register_slice_512} CONFIG.REG_AW {1} CONFIG.REG_AR {1} CONFIG.REG_B {1}] [get_ips axi_register_slice_512]


#####################################
# interconnect_3to1
create_bd_design "interconnect_3to1"
create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 axi_interconnect_3to1
set_property name interconnect_3to1 [get_bd_cells axi_interconnect_3to1]
set_property -dict [list CONFIG.NUM_SI {3} CONFIG.NUM_MI {1} CONFIG.ENABLE_ADVANCED_OPTIONS {1} CONFIG.S00_ARB_PRIORITY {15} CONFIG.S01_ARB_PRIORITY {15} CONFIG.NUM_MI {1}] [get_bd_cells interconnect_3to1]
create_bd_intf_port -mode Slave -vlnv xilinx.com:interface:aximm_rtl:1.0 S00_AXI_INSTR
set_property -dict [list CONFIG.ID_WIDTH [get_property CONFIG.ID_WIDTH [get_bd_intf_pins interconnect_3to1/xbar/S00_AXI]] CONFIG.HAS_REGION [get_property CONFIG.HAS_REGION [get_bd_intf_pins interconnect_3to1/xbar/S00_AXI]] CONFIG.NUM_READ_OUTSTANDING [get_property CONFIG.NUM_READ_OUTSTANDING [get_bd_intf_pins interconnect_3to1/xbar/S00_AXI]] CONFIG.NUM_WRITE_OUTSTANDING [get_property CONFIG.NUM_WRITE_OUTSTANDING [get_bd_intf_pins interconnect_3to1/xbar/S00_AXI]]] [get_bd_intf_ports S00_AXI_INSTR]
connect_bd_intf_net [get_bd_intf_pins interconnect_3to1/S00_AXI] [get_bd_intf_ports S00_AXI_INSTR]
create_bd_intf_port -mode Slave -vlnv xilinx.com:interface:aximm_rtl:1.0 S01_AXI_DATA
set_property -dict [list CONFIG.ID_WIDTH [get_property CONFIG.ID_WIDTH [get_bd_intf_pins interconnect_3to1/xbar/S01_AXI]] CONFIG.HAS_REGION [get_property CONFIG.HAS_REGION [get_bd_intf_pins interconnect_3to1/xbar/S01_AXI]] CONFIG.NUM_READ_OUTSTANDING [get_property CONFIG.NUM_READ_OUTSTANDING [get_bd_intf_pins interconnect_3to1/xbar/S01_AXI]] CONFIG.NUM_WRITE_OUTSTANDING [get_property CONFIG.NUM_WRITE_OUTSTANDING [get_bd_intf_pins interconnect_3to1/xbar/S01_AXI]]] [get_bd_intf_ports S01_AXI_DATA]
connect_bd_intf_net [get_bd_intf_pins interconnect_3to1/S01_AXI] [get_bd_intf_ports S01_AXI_DATA]
create_bd_intf_port -mode Slave -vlnv xilinx.com:interface:aximm_rtl:1.0 S02_AXI_DMA
set_property -dict [list CONFIG.ID_WIDTH [get_property CONFIG.ID_WIDTH [get_bd_intf_pins interconnect_3to1/xbar/S02_AXI]] CONFIG.HAS_REGION [get_property CONFIG.HAS_REGION [get_bd_intf_pins interconnect_3to1/xbar/S02_AXI]] CONFIG.NUM_READ_OUTSTANDING [get_property CONFIG.NUM_READ_OUTSTANDING [get_bd_intf_pins interconnect_3to1/xbar/S02_AXI]] CONFIG.NUM_WRITE_OUTSTANDING [get_property CONFIG.NUM_WRITE_OUTSTANDING [get_bd_intf_pins interconnect_3to1/xbar/S02_AXI]]] [get_bd_intf_ports S02_AXI_DMA]
connect_bd_intf_net [get_bd_intf_pins interconnect_3to1/S02_AXI] [get_bd_intf_ports S02_AXI_DMA]
create_bd_intf_port -mode Master -vlnv xilinx.com:interface:aximm_rtl:1.0 M00_AXI
set_property -dict [list CONFIG.NUM_READ_OUTSTANDING [get_property CONFIG.NUM_READ_OUTSTANDING [get_bd_intf_pins interconnect_3to1/xbar/M00_AXI]] CONFIG.NUM_WRITE_OUTSTANDING [get_property CONFIG.NUM_WRITE_OUTSTANDING [get_bd_intf_pins interconnect_3to1/xbar/M00_AXI]]] [get_bd_intf_ports M00_AXI]
connect_bd_intf_net [get_bd_intf_pins interconnect_3to1/M00_AXI] [get_bd_intf_ports M00_AXI]
connect_bd_net [get_bd_pins interconnect_3to1/ACLK] [get_bd_pins interconnect_3to1/S00_ACLK] -boundary_type upper
connect_bd_net [get_bd_pins interconnect_3to1/M00_ACLK] [get_bd_pins interconnect_3to1/ACLK] -boundary_type upper
connect_bd_net [get_bd_pins interconnect_3to1/S01_ACLK] [get_bd_pins interconnect_3to1/ACLK] -boundary_type upper
connect_bd_net [get_bd_pins interconnect_3to1/S02_ACLK] [get_bd_pins interconnect_3to1/ACLK] -boundary_type upper
create_bd_port -dir I -type clk -freq_hz 300000000 ACLK
connect_bd_net [get_bd_pins /interconnect_3to1/ACLK] [get_bd_ports ACLK]
create_bd_port -dir I -type rst ARESETN
connect_bd_net [get_bd_pins /interconnect_3to1/ARESETN] [get_bd_ports ARESETN]
connect_bd_net [get_bd_ports ARESETN] [get_bd_pins interconnect_3to1/S00_ARESETN]
connect_bd_net [get_bd_ports ARESETN] [get_bd_pins interconnect_3to1/M00_ARESETN]
connect_bd_net [get_bd_ports ARESETN] [get_bd_pins interconnect_3to1/S01_ARESETN]
connect_bd_net [get_bd_ports ARESETN] [get_bd_pins interconnect_3to1/S02_ARESETN]
set_property -dict [list CONFIG.NUM_WRITE_OUTSTANDING {32} CONFIG.NUM_READ_OUTSTANDING {32} CONFIG.READ_WRITE_MODE {READ_ONLY} CONFIG.ADDR_WIDTH {64} CONFIG.ID_WIDTH {6} CONFIG.FREQ_HZ {300000000} CONFIG.HAS_REGION {1}] [get_bd_intf_ports S00_AXI_INSTR]
set_property -dict [list CONFIG.NUM_WRITE_OUTSTANDING {32} CONFIG.NUM_READ_OUTSTANDING {32} CONFIG.READ_WRITE_MODE {READ_ONLY} CONFIG.ADDR_WIDTH {64} CONFIG.ID_WIDTH {6} CONFIG.FREQ_HZ {300000000} CONFIG.HAS_REGION {1}] [get_bd_intf_ports S01_AXI_DATA]
set_property -dict [list CONFIG.READ_WRITE_MODE {READ_WRITE} CONFIG.ID_WIDTH {6} CONFIG.DATA_WIDTH {512}] [get_bd_intf_ports S01_AXI_DATA]
set_property -dict [list CONFIG.NUM_WRITE_OUTSTANDING {32} CONFIG.NUM_READ_OUTSTANDING {32} CONFIG.ADDR_WIDTH {64} CONFIG.ID_WIDTH {6} CONFIG.FREQ_HZ {300000000} CONFIG.DATA_WIDTH {512} CONFIG.HAS_REGION {1}] [get_bd_intf_ports S02_AXI_DMA]
set_property -dict [list CONFIG.NUM_WRITE_OUTSTANDING {32} CONFIG.NUM_READ_OUTSTANDING {32} CONFIG.ADDR_WIDTH {64} CONFIG.FREQ_HZ {300000000} CONFIG.DATA_WIDTH {512}] [get_bd_intf_ports M00_AXI]
assign_bd_address
set_property offset 0x0000000000000000 [get_bd_addr_segs {S00_AXI_INSTR/SEG_M00_AXI_Reg}]
set_property range 1T [get_bd_addr_segs {S00_AXI_INSTR/SEG_M00_AXI_Reg}]
set_property offset 0x0000000000000000 [get_bd_addr_segs {S01_AXI_DATA/SEG_M00_AXI_Reg}]
set_property range 1T [get_bd_addr_segs {S01_AXI_DATA/SEG_M00_AXI_Reg}]
set_property offset 0x0000000000000000 [get_bd_addr_segs {S02_AXI_DMA/SEG_M00_AXI_Reg}]
set_property range 1T [get_bd_addr_segs {S02_AXI_DMA/SEG_M00_AXI_Reg}]
update_compile_order -fileset sources_1

set proj_name kernel_pack
set bd_path $path_to_tmp_project/$proj_name.srcs/sources_1/bd/interconnect_3to1
set gen_path $path_to_tmp_project/$proj_name.gen/sources_1/bd/interconnect_3to1
set ip_list [list interconnect_3to1_auto_us_0  interconnect_3to1_xbar_0 interconnect_3to1_axi_interconnect_3to1_0]
make_wrapper -files [get_files $bd_path/interconnect_3to1.bd] -top

if { [file exists $gen_path/synth/interconnect_3to1.v] } {
    file copy -force $gen_path/synth/interconnect_3to1.v $path_to_tmp_project
} elseif { [file exists $bd_path/synth/interconnect_3to1.v] } { 
    file copy -force $bd_path/synth/interconnect_3to1.v $path_to_tmp_project
}


foreach ip $ip_list {
    file mkdir $path_to_tmp_project/$ip 
    file copy -force $bd_path/ip/$ip/$ip.xci $path_to_tmp_project/$ip/
}

export_ip_user_files -of_objects  [get_files $bd_path/interconnect_3to1.bd] -no_script -reset -force -quiet
remove_files $bd_path/interconnect_3to1.bd

read_verilog -sv  $path_to_tmp_project/interconnect_3to1.v
foreach ip $ip_list { 
    read_ip [glob $path_to_tmp_project/$ip/$ip.xci ]
}


