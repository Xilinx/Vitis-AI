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

set path_to_packaged "./packaged_kernel/ip"
set path_to_tmp_project "./tmp_kernel_pack"

create_project -force kernel_pack $path_to_tmp_project -part xcu280-fsvh2892-2L-e
set_property ip_repo_paths "../../DPUCAHX8L_A_SRC/DPU" [current_project]
add_files $TOP

#=================================================================================================================
#use block design
set bd_design_name DPUCAHX8L_A_bd
create_bd_design ${bd_design_name}
  
  #global setting
  set name_rule [list VB_M_AXI_00 VB_M_AXI_01 VB_M_AXI_02 VB_M_AXI_03 VB_M_AXI_04 VB_M_AXI_05 VB_M_AXI_06 VB_M_AXI_07 SYS_M_AXI_00 SYS_M_AXI_01 SYS_M_AXI_02]
  #create all ports and relative clock converter
  set all_buses ""
  # Create interface ports
  foreach name $name_rule {
      set DPU_$name  [ create_bd_intf_port -mode Master -vlnv xilinx.com:interface:aximm_rtl:1.0 DPU_$name ]
      append all_buses "DPU_$name:"
      ## 
      set_property -dict [ list \
          CONFIG.ADDR_WIDTH {33} \
          CONFIG.DATA_WIDTH {256} \
          CONFIG.FREQ_HZ "[expr (100* 1000000)]" \
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
       ] [set DPU_$name]
       
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
   CONFIG.FREQ_HZ {100000000} \
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
  set_property -dict [ list \
   CONFIG.ASSOCIATED_BUSIF "$s_axi_control" \
  ] $ap_clk
  set ap_clk_2 [ create_bd_port -dir I -type clk ap_clk_2 ]
  set ap_rst_n [ create_bd_port -dir I -type rst ap_rst_n ]
  set ap_rst_n_2 [ create_bd_port -dir I -type rst ap_rst_n_2 ]
  set interrupt [ create_bd_port -dir O -from 0 -to 0 -type intr interrupt ]
  set_property -dict [ list \
   CONFIG.ASSOCIATED_BUSIF {} \
 ] $ap_clk_2

  # Create instance: axi_clock_converter_csr, and set properties
  set axi_clock_converter_csr [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_clock_converter:2.1 axi_clock_converter_csr ]

  # Create instance: dpu_clock_gen_0, and set properties
  #set dpu_clock_gen_0 [ create_bd_cell -type ip -vlnv deephi:user:dpu_clock_gen:1.0 dpu_clock_gen_0 ]

  # Create instance: dpu_top_0, and set properties
  set dpu_top_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:DPUCAHX8L:1.0 dpu_top_0 ]
  set_property -dict [list CONFIG.FREQ "[expr $ACLK_FREQ]"] [get_bd_cells dpu_top_0]

  # Create interface connections

  foreach name $name_rule {
      connect_bd_intf_net [get_bd_intf_ports DPU_$name]  [get_bd_intf_pins axi_clock_converter_$name/M_AXI]
      connect_bd_intf_net [get_bd_intf_pins axi_clock_converter_$name/S_AXI]  [get_bd_intf_pins dpu_top_0/DPU_$name]
      connect_bd_net [get_bd_ports ap_clk] [get_bd_pins axi_clock_converter_$name/m_axi_aclk]
      connect_bd_net [get_bd_ports ap_rst_n] [get_bd_pins axi_clock_converter_$name/m_axi_aresetn]
      connect_bd_net [get_bd_pins dpu_top_0/ACLK_OUT] [get_bd_pins axi_clock_converter_$name/s_axi_aclk]
      connect_bd_net [get_bd_pins dpu_top_0/ARESETN_OUT] [get_bd_pins axi_clock_converter_$name/s_axi_aresetn]
  }

  # Create port connections
  connect_bd_net [get_bd_ports ap_clk] [get_bd_pins axi_clock_converter_csr/s_axi_aclk]
  connect_bd_net [get_bd_ports ap_clk_2] [get_bd_pins dpu_top_0/refclk]
  connect_bd_net [get_bd_ports ap_rst_n] [get_bd_pins axi_clock_converter_csr/s_axi_aresetn]
  connect_bd_intf_net [get_bd_intf_ports s_axi_control]  [get_bd_intf_pins axi_clock_converter_csr/S_AXI]
  connect_bd_intf_net [get_bd_intf_pins axi_clock_converter_csr/M_AXI]  [get_bd_intf_pins dpu_top_0/DPU_CSR_S_AXI]
  connect_bd_net [get_bd_pins dpu_top_0/DPU_CSR_S_AXI_CLK] [get_bd_pins axi_clock_converter_csr/m_axi_aclk]
  connect_bd_net [get_bd_pins dpu_top_0/DPU_CSR_S_AXI_RST_N] [get_bd_pins axi_clock_converter_csr/m_axi_aresetn]
  connect_bd_net [get_bd_ports interrupt] [get_bd_pins dpu_top_0/dpu_interrupt]

  #if { [get_bd_pins dpu_top_0/reg_clk_throttle] != "" && [get_bd_pins dpu_clock_gen_0/dynclkthrottle] != ""} {
  #  connect_bd_net [get_bd_pins dpu_top_0/reg_clk_throttle] [get_bd_pins dpu_clock_gen_0/dynclkthrottle]
  #}

  # Create address segments
  # assign_bd_address [get_bd_addr_segs {dpu_top_0/s_axi_csr/reg0 }]

  foreach name $name_rule {
  	create_bd_addr_seg -range 0x000200000000 -offset 0x00000000 [get_bd_addr_spaces dpu_top_0/DPU_${name}] [get_bd_addr_segs DPU_${name}/Reg] SEG_DPU_${name}_Reg
  }
  create_bd_addr_seg -range 0x00001000 -offset 0x00000000 [get_bd_addr_spaces s_axi_control] [get_bd_addr_segs dpu_top_0/DPU_CSR_S_AXI/reg0] SEG_dpu_top_0_reg0
  
  validate_bd_design

  save_bd_design

  #generate_target all [get_files $path_to_tmp_project/kernel_pack.srcs/sources_1/bd/v3e_bd/v3e_bd.bd]
  #create_ip_run [get_files -of_objects [get_fileset sources_1] $path_to_tmp_project/kernel_pack.srcs/sources_1/bd/v3e_bd/v3e_bd.bd]
  #launch_runs -jobs 28 [get_runs {v3e_bd_axi_clock_converter_0_0_synth_1 v3e_bd_axi_clock_converter_1_0_synth_1 v3e_bd_axi_clock_converter_2_0_synth_1 v3e_bd_axi_clock_converter_I_0_synth_1 v3e_bd_axi_clock_converter_W0_0_synth_1 v3e_bd_axi_clock_converter_W1_0_synth_1 v3e_bd_axi_clock_converter_csr_0_synth_1 v3e_bd_dpu_clock_gen_0_0_synth_1 v3e_bd_dpu_top_0_0_synth_1}]
  generate_target {synthesis} [get_files $path_to_tmp_project/kernel_pack.srcs/sources_1/bd/${bd_design_name}/${bd_design_name}.bd]
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
#px::associate_bus_interfaces -busif DPU_$name -clock ap_clk [ipx::current_core]
ipx::associate_bus_interfaces -busif s_axi_control -clock ap_clk [ipx::current_core]
set_property xpm_libraries {XPM_CDC XPM_MEMORY XPM_FIFO} [ipx::current_core]
set_property supported_families { } [ipx::current_core]
set_property auto_family_support_level level_2 [ipx::current_core]
ipx::update_checksums [ipx::current_core]
ipx::save_core [ipx::current_core]
close_project -delete
