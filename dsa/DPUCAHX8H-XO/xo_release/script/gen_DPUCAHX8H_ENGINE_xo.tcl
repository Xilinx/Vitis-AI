# /*******************************************************************************
# Copyright (c) 2018, Xilinx, Inc.
# All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
# 
# 1. Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
# 
# 
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
# 
# 
# 3. Neither the name of the copyright holder nor the names of its contributors
# may be used to endorse or promote products derived from this software
# without specific prior written permission.
# 
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,THE IMPLIED 
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
# IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, 
# INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, 
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, 
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY 
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING 
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
# EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# *******************************************************************************/
puts "0,[lindex $::argv 0]"
puts "1,[lindex $::argv 1]"
puts "2,[lindex $::argv 2]"
#puts "3,[lindex $::argv 3]"
#puts "4,[lindex $::argv 4]"
#puts "5,[lindex $::argv 5]"
#puts "6,[lindex $::argv 6]"
#puts "7,[lindex $::argv 7]"
#puts "8,[lindex $::argv 8]"
#puts "9,[lindex $::argv 9]"
#puts "10,[lindex $::argv 10]"
#puts "11,[lindex $::argv 11]"
if { $::argc != 3 && $::argc != 2 } {
    puts "ERROR: Program \"$::argv0\" requires 3 guments,<xsa> <?ENGINE> <ACLK_FREQ>!\n"
    puts "Usage: $::argv0 <xoname> <krnl_name> <target> <device> <xsa> <DPU_SRC> <DPU_DIR> <ALVEO> <PART> <ACLK_FREQ> <AXI_SHARE>\n"
    exit
}
set SLR_N 	  [lindex $::argv 0]
set ACLK_FREQ [lindex $::argv 1]
set card       [lindex $::argv 2]
if {${card} == "u280"} {
set PART xcu280-fsvh2892-2L-e
} else {
set PART xcu50-fsvh2104-2L-e
}

if {${card} == "u280"} {
set xsa xilinx_u280_xdma_201920_3
} else {
set xsa xilinx_u50lv_gen3x4_xdma_2_202010_1
}

set xoname    DPUCAHX8H_${SLR_N}.xo
set krnl_name DPUCAHX8H_${SLR_N}
set target    hw
#set device    [lindex $::argv 3]

set DPU_SRC   ./src/DPUCAHX8H_${SLR_N}.v
#set DPU_DIR   [lindex $::argv 6]
#set ALVEO     [lindex $::argv 7]
set AXI_SHARE false

set suffix "${krnl_name}_${target}_${xsa}"

###begin source -notrace ./scripts/package_kernel_DPUCAHX8H_ENGINE.tcl
set path_to_packaged "./packaged_kernel/ip_${suffix}"
set path_to_tmp_project "./tmp_kernel_pack_${suffix}"

create_project -force kernel_pack $path_to_tmp_project -part $PART
set_property ip_repo_paths "../DPUCAHX8H_SRC/DPU/" [current_project]
add_files $DPU_SRC

#=================================================================================================================
#use block design
set bd_design_name v3e_bd
create_bd_design ${bd_design_name}
  
  #global setting
if { $SLR_N == "1ENGINE" } {
  if { $AXI_SHARE == "false" } {
      set name_rule [list 0 I0 W0 W1]
  } else {
      set name_rule [list 0 I0]
  }
}
if { $SLR_N == "2ENGINE" } {
  if { $AXI_SHARE == "false" } {
      set name_rule [list 0 1  I0 W0 W1]
  } else {
      set name_rule [list 0 1  I0]
  }
}
if { $SLR_N == "3ENGINE" } {
  if { $AXI_SHARE == "false" } {
      set name_rule [list 0 1 2 I0 W0 W1]
  } else {
      set name_rule [list 0 1 2 I0]
  }
}
if { $SLR_N == "4ENGINE" } {
  if { $AXI_SHARE == "false" } {
      set name_rule [list 0 1 2 3 I0 W0 W1]
  } else {
      set name_rule [list 0 1 2 3 I0]
  }
}
if { $SLR_N == "5ENGINE" } {
  if { $AXI_SHARE == "false" } {
      set name_rule [list 0 1 2 3 4 I0 W0 W1]
  } else {
      set name_rule [list 0 1 2 3 4 I0]
  }
}
if { $SLR_N == "6ENGINE" } {
  if { $AXI_SHARE == "false" } {
      set name_rule [list 0 1 2 3 4 6 I0 W0 W1]
  } else {
      set name_rule [list 0 1 2 3 4 6 I0]
  }
}

  #create all ports and relative clock converter
  set all_buses ""
  # Create interface ports
  foreach name $name_rule {
      set DPU_AXI_$name  [ create_bd_intf_port -mode Master -vlnv xilinx.com:interface:aximm_rtl:1.0 DPU_AXI_$name ]
      append all_buses "DPU_AXI_$name:"
      ## 
      set_property -dict [ list \
          CONFIG.ADDR_WIDTH {33} \
          CONFIG.DATA_WIDTH {256} \
          CONFIG.HAS_BRESP {0} \
          CONFIG.HAS_BURST {0} \
          CONFIG.HAS_CACHE {0} \
          CONFIG.HAS_LOCK {0} \
          CONFIG.HAS_PROT {0} \
          CONFIG.HAS_QOS {0} \
          CONFIG.HAS_REGION {0} \
          CONFIG.HAS_WSTRB {1} \
          CONFIG.NUM_READ_OUTSTANDING {1} \
          CONFIG.NUM_WRITE_OUTSTANDING {1} \
          CONFIG.PROTOCOL {AXI3} \
          CONFIG.READ_WRITE_MODE {READ_WRITE} \
       ] [set DPU_AXI_$name]
       
       # Create instance: axi_clock_converter_0, and set properties
       set axi_clock_converter_$name [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_clock_converter:2.1 axi_clock_converter_$name ]
  }

  set s_axi_control [ create_bd_intf_port -mode Slave -vlnv xilinx.com:interface:aximm_rtl:1.0 s_axi_control ]
  append all_buses "s_axi_control"
  set_property -dict [ list \
   CONFIG.ADDR_WIDTH {12} \
   CONFIG.ARUSER_WIDTH {0} \
   CONFIG.AWUSER_WIDTH {0} \
   CONFIG.BUSER_WIDTH {0} \
   CONFIG.DATA_WIDTH {32} \
   CONFIG.HAS_BRESP {1} \
   CONFIG.HAS_BURST {0} \
   CONFIG.HAS_CACHE {0} \
   CONFIG.HAS_LOCK {0} \
   CONFIG.HAS_PROT {0} \
   CONFIG.HAS_QOS {0} \
   CONFIG.HAS_REGION {0} \
   CONFIG.HAS_RRESP {1} \
   CONFIG.HAS_WSTRB {1} \
   CONFIG.ID_WIDTH {0} \
   CONFIG.MAX_BURST_LENGTH {1} \
   CONFIG.NUM_READ_OUTSTANDING {1} \
   CONFIG.NUM_READ_THREADS {1} \
   CONFIG.NUM_WRITE_OUTSTANDING {1} \
   CONFIG.NUM_WRITE_THREADS {1} \
   CONFIG.PROTOCOL {AXI4LITE} \
   CONFIG.READ_WRITE_MODE {READ_WRITE} \
   CONFIG.RUSER_BITS_PER_BYTE {0} \
   CONFIG.RUSER_WIDTH {0} \
   CONFIG.SUPPORTS_NARROW_BURST {0} \
   CONFIG.WUSER_BITS_PER_BYTE {0} \
   CONFIG.WUSER_WIDTH {0} \
   ] $s_axi_control

    # Create ports
  set ap_clk [ create_bd_port -dir I -type clk ap_clk ]
  set_property -dict [ list \
   CONFIG.ASSOCIATED_BUSIF "$all_buses" \
 ] $ap_clk

  set ap_clk_2 [ create_bd_port -dir I -type clk ap_clk_2 ]
  set ap_rst_n [ create_bd_port -dir I -type rst ap_rst_n ]
  set ap_rst_n_2 [ create_bd_port -dir I -type rst ap_rst_n_2 ]
  set interrupt [ create_bd_port -dir O -from 0 -to 0 -type intr interrupt ]


  # Create instance: axi_clock_converter_csr, and set properties
  set axi_clock_converter_csr [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_clock_converter:2.1 axi_clock_converter_csr ]

  # Create instance: dpu_top_0, and set properties
  set dpu_top_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:dpu_top:1.0 dpu_top_0 ]
  set_property -dict [list CONFIG.DPU_NUM [string index $SLR_N 0] CONFIG.FREQ $ACLK_FREQ] [get_bd_cells dpu_top_0]

if {${card} == "u280"} {
set_property -dict [list CONFIG.FM_BKG_MERGE {1}] [get_bd_cells dpu_top_0]
set_property -dict [list CONFIG.FM_LOAD_PER_CORE {1} CONFIG.SAVE_PER_ENGINE {1}] [get_bd_cells dpu_top_0]
set_property -dict [list CONFIG.CLK_GATE_EN {1}] [get_bd_cells dpu_top_0]
} elseif  {${card} == "u50lv9e"} {
set_property -dict [list CONFIG.CLK_GATE_EN {0}] [get_bd_cells dpu_top_0]
} elseif  {${card} == "u50lv"} {
set_property -dict [list CONFIG.CLK_GATE_EN {0}] [get_bd_cells dpu_top_0]
set_property -dict [list CONFIG.FM_LOAD_PER_CORE {1} CONFIG.SAVE_PER_ENGINE {1}] [get_bd_cells dpu_top_0]
} elseif  {${card} == "u50"} {
set_property -dict [list CONFIG.MISC_PP_N {2}] [get_bd_cells dpu_top_0]
}

  # Create interface connections

  foreach name $name_rule {
      connect_bd_intf_net [get_bd_intf_ports DPU_AXI_$name]  [get_bd_intf_pins axi_clock_converter_$name/M_AXI]
      connect_bd_intf_net [get_bd_intf_pins axi_clock_converter_$name/S_AXI]  [get_bd_intf_pins dpu_top_0/DPU_AXI_$name]
      connect_bd_net [get_bd_ports ap_clk] [get_bd_pins axi_clock_converter_$name/m_axi_aclk]
      connect_bd_net [get_bd_ports ap_rst_n] [get_bd_pins axi_clock_converter_$name/m_axi_aresetn]
      connect_bd_net [get_bd_pins axi_clock_converter_$name/s_axi_aclk] [get_bd_pins dpu_top_0/ACLK_OUT]
      connect_bd_net [get_bd_pins axi_clock_converter_$name/s_axi_aresetn] [get_bd_pins dpu_top_0/ARESETN_OUT]
  }

  # Create port connections
  connect_bd_net [get_bd_ports ap_clk] [get_bd_pins axi_clock_converter_csr/s_axi_aclk]
  connect_bd_net [get_bd_ports ap_clk_2] [get_bd_pins dpu_top_0/refclk]
  connect_bd_net [get_bd_ports ap_rst_n] [get_bd_pins axi_clock_converter_csr/s_axi_aresetn]

  connect_bd_intf_net [get_bd_intf_ports s_axi_control]  [get_bd_intf_pins axi_clock_converter_csr/S_AXI]
  connect_bd_intf_net [get_bd_intf_pins axi_clock_converter_csr/M_AXI]  [get_bd_intf_pins dpu_top_0/s_axi_csr]
  connect_bd_net [get_bd_pins axi_clock_converter_csr/m_axi_aclk] [get_bd_pins dpu_top_0/s_axi_clk]
  connect_bd_net [get_bd_pins axi_clock_converter_csr/m_axi_aresetn] [get_bd_pins dpu_top_0/s_axi_rst_n]

  connect_bd_net [get_bd_ports interrupt] [get_bd_pins dpu_top_0/dpu_interrupt]

  # Create address segments
  # assign_bd_address [get_bd_addr_segs {dpu_top_0/s_axi_csr/reg0 }]

  foreach name $name_rule {
  	create_bd_addr_seg -range 0x000200000000 -offset 0x00000000 [get_bd_addr_spaces dpu_top_0/DPU_AXI_${name}] [get_bd_addr_segs DPU_AXI_${name}/Reg] SEG_DPU_AXI_${name}_Reg
  }
  create_bd_addr_seg -range 0x00001000 -offset 0x00000000 [get_bd_addr_spaces s_axi_control] [get_bd_addr_segs dpu_top_0/s_axi_csr/reg0] SEG_dpu_top_0_reg0
  
  validate_bd_design

  save_bd_design

  #generate_target all [get_files $path_to_tmp_project/kernel_pack.srcs/sources_1/bd/v3e_bd/v3e_bd.bd]
  #create_ip_run [get_files -of_objects [get_fileset sources_1] $path_to_tmp_project/kernel_pack.srcs/sources_1/bd/v3e_bd/v3e_bd.bd]
  #launch_runs -jobs 28 [get_runs {v3e_bd_axi_clock_converter_0_0_synth_1 v3e_bd_axi_clock_converter_1_0_synth_1 v3e_bd_axi_clock_converter_2_0_synth_1 v3e_bd_axi_clock_converter_I_0_synth_1 v3e_bd_axi_clock_converter_W0_0_synth_1 v3e_bd_axi_clock_converter_W1_0_synth_1 v3e_bd_axi_clock_converter_csr_0_synth_1 v3e_bd_dpu_top_0_0_synth_1}]
  generate_target {synthesis} [get_files $path_to_tmp_project/kernel_pack.srcs/sources_1/bd/v3e_bd/v3e_bd.bd]
#=================================================================================================================




update_compile_order -fileset sources_1
update_compile_order -fileset sim_1
ipx::package_project -root_dir $path_to_packaged -vendor xilinx.com -library RTLKernel -taxonomy /KernelIP -import_files -set_current false
ipx::unload_core $path_to_packaged/component.xml
ipx::edit_ip_in_project -upgrade true -name tmp_edit_project -directory $path_to_packaged $path_to_packaged/component.xml
set_property core_revision 2 [ipx::current_core]
foreach up [ipx::get_user_parameters] {
  ipx::remove_user_parameter [get_property NAME $up] [ipx::current_core]
}
set_property sdx_kernel true [ipx::current_core]
set_property sdx_kernel_type rtl [ipx::current_core]
ipx::create_xgui_files [ipx::current_core]
#ipx::associate_bus_interfaces -busif m_axi_gmem -clock ap_clk [ipx::current_core]
#ipx::associate_bus_interfaces -busif s_axi_control -clock ap_clk [ipx::current_core]
set_property xpm_libraries {XPM_CDC XPM_MEMORY XPM_FIFO} [ipx::current_core]
set_property supported_families { } [ipx::current_core]
set_property auto_family_support_level level_2 [ipx::current_core]
ipx::update_checksums [ipx::current_core]
ipx::save_core [ipx::current_core]
close_project -delete
###end source -notrace ./scripts/package_kernel_DPUCAHX8H_ENGINE.tcl

if {[file exists "${xoname}"]} {
    file delete -force "${xoname}"
}

package_xo -xo_path ${xoname} -kernel_name DPUCAHX8H_${SLR_N} -ip_directory ./packaged_kernel/ip_${suffix} -kernel_xml ./src/kernel_DPUCAHX8H_${SLR_N}.xml
exit
