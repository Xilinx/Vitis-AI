
################################################################
# This is a generated script based on design: vck190_es1_base_trd_platform1
#
# Though there are limitations about the generated script,
# the main purpose of this utility is to make learning
# IP Integrator Tcl commands easier.
################################################################

namespace eval _tcl {
proc get_script_folder {} {
   set script_path [file normalize [info script]]
   set script_folder [file dirname $script_path]
   return $script_folder
}
}
variable script_folder
set script_folder [_tcl::get_script_folder]

################################################################
# Check if script is running in correct Vivado version.
################################################################
set scripts_vivado_version 2020.2
set current_vivado_version [version -short]

if { [string first $scripts_vivado_version $current_vivado_version] == -1 } {
   puts ""
   catch {common::send_gid_msg -ssname BD::TCL -id 2041 -severity "ERROR" "This script was generated using Vivado <$scripts_vivado_version> and is being run in <$current_vivado_version> of Vivado. Please run the script in Vivado <$scripts_vivado_version> then open the design in Vivado <$current_vivado_version>. Upgrade the design by running \"Tools => Report => Report IP Status...\", then run write_bd_tcl to create an updated script."}

   return 1
}

################################################################
# START
################################################################

# To test this script, run the following commands from Vivado Tcl console:
# source vck190_es1_base_trd_platform1_script.tcl

# If there is no project opened, this script will create a
# project, but make sure you do not have an existing project
# <./myproj/project_1.xpr> in the current working folder.

set list_projs [get_projects -quiet]
if { $list_projs eq "" } {
   create_project project_1 myproj -part xcvc1902-vsva2197-2MP-e-S-es1
}


# CHANGE DESIGN NAME HERE
variable design_name
set design_name vck190_es1_base_trd_platform1

# If you do not already have an existing IP Integrator design open,
# you can create a design using the following command:
#    create_bd_design $design_name

# Creating design if needed
set errMsg ""
set nRet 0

set cur_design [current_bd_design -quiet]
set list_cells [get_bd_cells -quiet]

if { ${design_name} eq "" } {
   # USE CASES:
   #    1) Design_name not set

   set errMsg "Please set the variable <design_name> to a non-empty value."
   set nRet 1

} elseif { ${cur_design} ne "" && ${list_cells} eq "" } {
   # USE CASES:
   #    2): Current design opened AND is empty AND names same.
   #    3): Current design opened AND is empty AND names diff; design_name NOT in project.
   #    4): Current design opened AND is empty AND names diff; design_name exists in project.

   if { $cur_design ne $design_name } {
      common::send_gid_msg -ssname BD::TCL -id 2001 -severity "INFO" "Changing value of <design_name> from <$design_name> to <$cur_design> since current design is empty."
      set design_name [get_property NAME $cur_design]
   }
   common::send_gid_msg -ssname BD::TCL -id 2002 -severity "INFO" "Constructing design in IPI design <$cur_design>..."

} elseif { ${cur_design} ne "" && $list_cells ne "" && $cur_design eq $design_name } {
   # USE CASES:
   #    5) Current design opened AND has components AND same names.

   set errMsg "Design <$design_name> already exists in your project, please set the variable <design_name> to another value."
   set nRet 1
} elseif { [get_files -quiet ${design_name}.bd] ne "" } {
   # USE CASES: 
   #    6) Current opened design, has components, but diff names, design_name exists in project.
   #    7) No opened design, design_name exists in project.

   set errMsg "Design <$design_name> already exists in your project, please set the variable <design_name> to another value."
   set nRet 2

} else {
   # USE CASES:
   #    8) No opened design, design_name not in project.
   #    9) Current opened design, has components, but diff names, design_name not in project.

   common::send_gid_msg -ssname BD::TCL -id 2003 -severity "INFO" "Currently there is no design <$design_name> in project, so creating one..."

   create_bd_design $design_name

   common::send_gid_msg -ssname BD::TCL -id 2004 -severity "INFO" "Making design <$design_name> as current_bd_design."
   current_bd_design $design_name

}

common::send_gid_msg -ssname BD::TCL -id 2005 -severity "INFO" "Currently the variable <design_name> is equal to \"$design_name\"."

if { $nRet != 0 } {
   catch {common::send_gid_msg -ssname BD::TCL -id 2006 -severity "ERROR" $errMsg}
   return $nRet
}

set bCheckIPsPassed 1
##################################################################
# CHECK IPs
##################################################################
set bCheckIPs 1
if { $bCheckIPs == 1 } {
   set list_check_ips "\ 
xilinx.com:ip:versal_cips:2.1\
xilinx.com:ip:axi_noc:1.0\
xilinx.com:ip:ai_engine:2.0\
xilinx.com:ip:axi_intc:4.1\
xilinx.com:ip:axi_perf_mon:5.0\
xilinx.com:ip:axi_vip:1.1\
xilinx.com:ip:clk_wizard:1.0\
xilinx.com:ip:proc_sys_reset:5.0\
xilinx.com:ip:smartconnect:1.0\
xilinx.com:ip:audio_formatter:1.0\
xilinx.com:user:hdmi_acr_ctrl:1.1\
xilinx.com:ip:xlconstant:1.1\
xilinx.com:ip:v_hdmi_tx_ss:3.1\
xilinx.com:ip:v_mix:5.1\
xilinx.com:ip:xlslice:1.0\
xilinx.com:ip:util_ds_buf:2.1\
xilinx.com:ip:axi_iic:2.0\
xilinx.com:ip:hdmi_gt_controller:1.0\
xilinx.com:ip:axis_register_slice:1.1\
xilinx.com:hls:ISPPipeline_accel:1.0\
xilinx.com:ip:axis_data_fifo:2.0\
xilinx.com:ip:axis_subset_converter:1.1\
xilinx.com:ip:v_frmbuf_wr:2.2\
xilinx.com:ip:v_proc_ss:2.3\
xilinx.com:ip:mipi_csi2_rx_subsystem:5.1\
xilinx.com:ip:bufg_gt:1.0\
xilinx.com:ip:gt_quad_base:1.1\
"

   set list_ips_missing ""
   common::send_gid_msg -ssname BD::TCL -id 2011 -severity "INFO" "Checking if the following IPs exist in the project's IP catalog: $list_check_ips ."

   foreach ip_vlnv $list_check_ips {
      set ip_obj [get_ipdefs -all $ip_vlnv]
      if { $ip_obj eq "" } {
         lappend list_ips_missing $ip_vlnv
      }
   }

   if { $list_ips_missing ne "" } {
      catch {common::send_gid_msg -ssname BD::TCL -id 2012 -severity "ERROR" "The following IPs are not found in the IP Catalog:\n  $list_ips_missing\n\nResolution: Please add the repository containing the IP(s) to the project." }
      set bCheckIPsPassed 0
   }

}

if { $bCheckIPsPassed != 1 } {
  common::send_gid_msg -ssname BD::TCL -id 2023 -severity "WARNING" "Will not continue with creation of design due to the error(s) above."
  return 3
}

##################################################################
# DESIGN PROCs
##################################################################


# Hierarchical cell: gt_refclk1
proc create_hier_cell_gt_refclk1 { parentCell nameHier } {

  variable script_folder

  if { $parentCell eq "" || $nameHier eq "" } {
     catch {common::send_gid_msg -ssname BD::TCL -id 2092 -severity "ERROR" "create_hier_cell_gt_refclk1() - Empty argument(s)!"}
     return
  }

  # Get object for parentCell
  set parentObj [get_bd_cells $parentCell]
  if { $parentObj == "" } {
     catch {common::send_gid_msg -ssname BD::TCL -id 2090 -severity "ERROR" "Unable to find parent cell <$parentCell>!"}
     return
  }

  # Make sure parentObj is hier blk
  set parentType [get_property TYPE $parentObj]
  if { $parentType ne "hier" } {
     catch {common::send_gid_msg -ssname BD::TCL -id 2091 -severity "ERROR" "Parent <$parentObj> has TYPE = <$parentType>. Expected to be <hier>."}
     return
  }

  # Save current instance; Restore later
  set oldCurInst [current_bd_instance .]

  # Set parent object as current
  current_bd_instance $parentObj

  # Create cell and set as current instance
  set hier_obj [create_bd_cell -type hier $nameHier]
  current_bd_instance $hier_obj

  # Create interface pins

  # Create pins
  create_bd_pin -dir I -from 0 -to 0 -type clk CLK_N_IN
  create_bd_pin -dir I -from 0 -to 0 -type clk CLK_P_IN
  create_bd_pin -dir O -from 0 -to 0 -type clk O
  create_bd_pin -dir O -from 0 -to 0 -type clk ODIV2

  # Create instance: dru_ibufds_gt_odiv2, and set properties
  set dru_ibufds_gt_odiv2 [ create_bd_cell -type ip -vlnv xilinx.com:ip:util_ds_buf:2.1 dru_ibufds_gt_odiv2 ]
  set_property -dict [ list \
   CONFIG.C_BUF_TYPE {BUFG_GT} \
 ] $dru_ibufds_gt_odiv2

  # Create instance: gt_refclk_buf, and set properties
  set gt_refclk_buf [ create_bd_cell -type ip -vlnv xilinx.com:ip:util_ds_buf:2.1 gt_refclk_buf ]
  set_property -dict [ list \
   CONFIG.C_BUF_TYPE {IBUFDSGTE} \
 ] $gt_refclk_buf

  # Create instance: vcc_const0, and set properties
  set vcc_const0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:xlconstant:1.1 vcc_const0 ]
  set_property -dict [ list \
   CONFIG.CONST_VAL {1} \
 ] $vcc_const0

  # Create port connections
  connect_bd_net -net HDMI_RX_CLK_N_IN_1 [get_bd_pins CLK_N_IN] [get_bd_pins gt_refclk_buf/IBUF_DS_N]
  connect_bd_net -net HDMI_RX_CLK_P_IN_1 [get_bd_pins CLK_P_IN] [get_bd_pins gt_refclk_buf/IBUF_DS_P]
  connect_bd_net -net dru_ibufds_gt_odiv2_BUFG_GT_O [get_bd_pins ODIV2] [get_bd_pins dru_ibufds_gt_odiv2/BUFG_GT_O]
  connect_bd_net -net gt_refclk_buf_IBUF_OUT [get_bd_pins O] [get_bd_pins gt_refclk_buf/IBUF_OUT]
  connect_bd_net -net net_gt_refclk_buf_IBUF_DS_ODIV2 [get_bd_pins dru_ibufds_gt_odiv2/BUFG_GT_I] [get_bd_pins gt_refclk_buf/IBUF_DS_ODIV2]
  connect_bd_net -net net_vcc_const0_dout [get_bd_pins dru_ibufds_gt_odiv2/BUFG_GT_CE] [get_bd_pins vcc_const0/dout]

  # Restore current instance
  current_bd_instance $oldCurInst
}

# Hierarchical cell: GT_Quad_and_Clk
proc create_hier_cell_GT_Quad_and_Clk { parentCell nameHier } {

  variable script_folder

  if { $parentCell eq "" || $nameHier eq "" } {
     catch {common::send_gid_msg -ssname BD::TCL -id 2092 -severity "ERROR" "create_hier_cell_GT_Quad_and_Clk() - Empty argument(s)!"}
     return
  }

  # Get object for parentCell
  set parentObj [get_bd_cells $parentCell]
  if { $parentObj == "" } {
     catch {common::send_gid_msg -ssname BD::TCL -id 2090 -severity "ERROR" "Unable to find parent cell <$parentCell>!"}
     return
  }

  # Make sure parentObj is hier blk
  set parentType [get_property TYPE $parentObj]
  if { $parentType ne "hier" } {
     catch {common::send_gid_msg -ssname BD::TCL -id 2091 -severity "ERROR" "Parent <$parentObj> has TYPE = <$parentType>. Expected to be <hier>."}
     return
  }

  # Save current instance; Restore later
  set oldCurInst [current_bd_instance .]

  # Set parent object as current
  current_bd_instance $parentObj

  # Create cell and set as current instance
  set hier_obj [create_bd_cell -type hier $nameHier]
  current_bd_instance $hier_obj

  # Create interface pins
  create_bd_intf_pin -mode Slave -vlnv xilinx.com:interface:gt_channel_debug_rtl:1.0 CH0_DEBUG

  create_bd_intf_pin -mode Slave -vlnv xilinx.com:interface:gt_channel_debug_rtl:1.0 CH1_DEBUG

  create_bd_intf_pin -mode Slave -vlnv xilinx.com:interface:gt_channel_debug_rtl:1.0 CH2_DEBUG

  create_bd_intf_pin -mode Slave -vlnv xilinx.com:interface:gt_channel_debug_rtl:1.0 CH3_DEBUG

  create_bd_intf_pin -mode Slave -vlnv xilinx.com:interface:gt_debug_rtl:1.0 GT_DEBUG

  create_bd_intf_pin -mode Slave -vlnv xilinx.com:interface:gt_tx_interface_rtl:1.0 TX0_GT_IP_Interface

  create_bd_intf_pin -mode Slave -vlnv xilinx.com:interface:gt_tx_interface_rtl:1.0 TX1_GT_IP_Interface

  create_bd_intf_pin -mode Slave -vlnv xilinx.com:interface:gt_tx_interface_rtl:1.0 TX2_GT_IP_Interface

  create_bd_intf_pin -mode Slave -vlnv xilinx.com:interface:gt_tx_interface_rtl:1.0 TX3_GT_IP_Interface


  # Create pins
  create_bd_pin -dir I -type clk GT_REFCLK1
  create_bd_pin -dir I -from 3 -to 0 RX_DATA_IN_rxn
  create_bd_pin -dir I -from 3 -to 0 RX_DATA_IN_rxp
  create_bd_pin -dir O -from 3 -to 0 TX_DATA_OUT_txn
  create_bd_pin -dir O -from 3 -to 0 TX_DATA_OUT_txp
  create_bd_pin -dir I altclk
  create_bd_pin -dir O ch0_iloresetdone
  create_bd_pin -dir O ch1_iloresetdone
  create_bd_pin -dir O ch2_iloresetdone
  create_bd_pin -dir O ch3_iloresetdone
  create_bd_pin -dir O gtpowergood
  create_bd_pin -dir O hsclk0_lcplllock
  create_bd_pin -dir I -type rst hsclk0_lcpllreset
  create_bd_pin -dir O hsclk1_lcplllock
  create_bd_pin -dir I -type rst hsclk1_lcpllreset
  create_bd_pin -dir O -type gt_usrclk tx_usrclk

  # Create instance: bufg_gt_tx, and set properties
  set bufg_gt_tx [ create_bd_cell -type ip -vlnv xilinx.com:ip:bufg_gt:1.0 bufg_gt_tx ]

  # Create instance: gt_quad_base_1, and set properties
  set gt_quad_base_1 [ create_bd_cell -type ip -vlnv xilinx.com:ip:gt_quad_base:1.1 gt_quad_base_1 ]
  set_property -dict [ list \
   CONFIG.REFCLK_STRING { \
     HSCLK0_LCPLLGTREFCLK1 refclk_PROT0_R1_multiple_ext_freq \
     HSCLK1_LCPLLGTREFCLK1 refclk_PROT0_R1_multiple_ext_freq \
   } \
 ] $gt_quad_base_1

  # Create instance: xlconstant_0, and set properties
  set xlconstant_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:xlconstant:1.1 xlconstant_0 ]
  set_property -dict [ list \
   CONFIG.CONST_VAL {0} \
 ] $xlconstant_0

  # Create interface connections
  connect_bd_intf_net -intf_net CH0_DEBUG_1 [get_bd_intf_pins CH0_DEBUG] [get_bd_intf_pins gt_quad_base_1/CH0_DEBUG]
  connect_bd_intf_net -intf_net CH1_DEBUG_1 [get_bd_intf_pins CH1_DEBUG] [get_bd_intf_pins gt_quad_base_1/CH1_DEBUG]
  connect_bd_intf_net -intf_net CH2_DEBUG_1 [get_bd_intf_pins CH2_DEBUG] [get_bd_intf_pins gt_quad_base_1/CH2_DEBUG]
  connect_bd_intf_net -intf_net CH3_DEBUG_1 [get_bd_intf_pins CH3_DEBUG] [get_bd_intf_pins gt_quad_base_1/CH3_DEBUG]
  connect_bd_intf_net -intf_net GT_DEBUG_1 [get_bd_intf_pins GT_DEBUG] [get_bd_intf_pins gt_quad_base_1/GT_DEBUG]
  connect_bd_intf_net -intf_net TX0_GT_IP_Interface_1 [get_bd_intf_pins TX0_GT_IP_Interface] [get_bd_intf_pins gt_quad_base_1/TX0_GT_IP_Interface]
  connect_bd_intf_net -intf_net TX1_GT_IP_Interface_1 [get_bd_intf_pins TX1_GT_IP_Interface] [get_bd_intf_pins gt_quad_base_1/TX1_GT_IP_Interface]
  connect_bd_intf_net -intf_net TX2_GT_IP_Interface_1 [get_bd_intf_pins TX2_GT_IP_Interface] [get_bd_intf_pins gt_quad_base_1/TX2_GT_IP_Interface]
  connect_bd_intf_net -intf_net TX3_GT_IP_Interface_1 [get_bd_intf_pins TX3_GT_IP_Interface] [get_bd_intf_pins gt_quad_base_1/TX3_GT_IP_Interface]

  # Create port connections
  connect_bd_net -net GT_REFCLK1_1 [get_bd_pins GT_REFCLK1] [get_bd_pins gt_quad_base_1/GT_REFCLK0]
  connect_bd_net -net RX_DATA_IN_rxn [get_bd_pins RX_DATA_IN_rxn] [get_bd_pins gt_quad_base_1/rxn]
  connect_bd_net -net RX_DATA_IN_rxp [get_bd_pins RX_DATA_IN_rxp] [get_bd_pins gt_quad_base_1/rxp]
  connect_bd_net -net altclk_1 [get_bd_pins altclk] [get_bd_pins gt_quad_base_1/altclk] [get_bd_pins gt_quad_base_1/apb3clk]
  connect_bd_net -net bufg_gt_1_usrclk [get_bd_pins tx_usrclk] [get_bd_pins bufg_gt_tx/usrclk] [get_bd_pins gt_quad_base_1/ch0_txusrclk] [get_bd_pins gt_quad_base_1/ch1_txusrclk] [get_bd_pins gt_quad_base_1/ch2_txusrclk] [get_bd_pins gt_quad_base_1/ch3_txusrclk]
  connect_bd_net -net gt_quad_base_1_ch0_iloresetdone [get_bd_pins ch0_iloresetdone] [get_bd_pins gt_quad_base_1/ch0_iloresetdone]
  connect_bd_net -net gt_quad_base_1_ch0_txoutclk [get_bd_pins bufg_gt_tx/outclk] [get_bd_pins gt_quad_base_1/ch0_txoutclk]
  connect_bd_net -net gt_quad_base_1_ch1_iloresetdone [get_bd_pins ch1_iloresetdone] [get_bd_pins gt_quad_base_1/ch1_iloresetdone]
  connect_bd_net -net gt_quad_base_1_ch2_iloresetdone [get_bd_pins ch2_iloresetdone] [get_bd_pins gt_quad_base_1/ch2_iloresetdone]
  connect_bd_net -net gt_quad_base_1_ch3_iloresetdone [get_bd_pins ch3_iloresetdone] [get_bd_pins gt_quad_base_1/ch3_iloresetdone]
  connect_bd_net -net gt_quad_base_1_gtpowergood [get_bd_pins gtpowergood] [get_bd_pins gt_quad_base_1/gtpowergood]
  connect_bd_net -net gt_quad_base_1_hsclk0_lcplllock [get_bd_pins hsclk0_lcplllock] [get_bd_pins gt_quad_base_1/hsclk0_lcplllock]
  connect_bd_net -net gt_quad_base_1_hsclk1_lcplllock [get_bd_pins hsclk1_lcplllock] [get_bd_pins gt_quad_base_1/hsclk1_lcplllock]
  connect_bd_net -net gt_quad_base_1_txn [get_bd_pins TX_DATA_OUT_txn] [get_bd_pins gt_quad_base_1/txn]
  connect_bd_net -net gt_quad_base_1_txp [get_bd_pins TX_DATA_OUT_txp] [get_bd_pins gt_quad_base_1/txp]
  connect_bd_net -net hsclk0_lcpllreset_1 [get_bd_pins hsclk0_lcpllreset] [get_bd_pins gt_quad_base_1/hsclk0_lcpllreset]
  connect_bd_net -net hsclk1_lcpllreset_1 [get_bd_pins hsclk1_lcpllreset] [get_bd_pins gt_quad_base_1/hsclk1_lcpllreset]
  connect_bd_net -net xlconstant_0_dout [get_bd_pins gt_quad_base_1/ch0_rxusrclk] [get_bd_pins gt_quad_base_1/ch1_rxusrclk] [get_bd_pins gt_quad_base_1/ch2_rxusrclk] [get_bd_pins gt_quad_base_1/ch3_rxusrclk] [get_bd_pins xlconstant_0/dout]

  # Restore current instance
  current_bd_instance $oldCurInst
}

# Hierarchical cell: mipi_csi_rx_ss
proc create_hier_cell_mipi_csi_rx_ss { parentCell nameHier } {

  variable script_folder

  if { $parentCell eq "" || $nameHier eq "" } {
     catch {common::send_gid_msg -ssname BD::TCL -id 2092 -severity "ERROR" "create_hier_cell_mipi_csi_rx_ss() - Empty argument(s)!"}
     return
  }

  # Get object for parentCell
  set parentObj [get_bd_cells $parentCell]
  if { $parentObj == "" } {
     catch {common::send_gid_msg -ssname BD::TCL -id 2090 -severity "ERROR" "Unable to find parent cell <$parentCell>!"}
     return
  }

  # Make sure parentObj is hier blk
  set parentType [get_property TYPE $parentObj]
  if { $parentType ne "hier" } {
     catch {common::send_gid_msg -ssname BD::TCL -id 2091 -severity "ERROR" "Parent <$parentObj> has TYPE = <$parentType>. Expected to be <hier>."}
     return
  }

  # Save current instance; Restore later
  set oldCurInst [current_bd_instance .]

  # Set parent object as current
  current_bd_instance $parentObj

  # Create cell and set as current instance
  set hier_obj [create_bd_cell -type hier $nameHier]
  current_bd_instance $hier_obj

  # Create interface pins
  create_bd_intf_pin -mode Master -vlnv xilinx.com:interface:iic_rtl:1.0 IIC_sensor

  create_bd_intf_pin -mode Slave -vlnv xilinx.com:interface:aximm_rtl:1.0 S_AXI

  create_bd_intf_pin -mode Slave -vlnv xilinx.com:interface:aximm_rtl:1.0 csirxss_s_axi

  create_bd_intf_pin -mode Slave -vlnv xilinx.com:interface:mipi_phy_rtl:1.0 mipi_phy_csi

  create_bd_intf_pin -mode Master -vlnv xilinx.com:interface:axis_rtl:1.0 video_out


  # Create pins
  create_bd_pin -dir O -type intr csirxss_csi_irq
  create_bd_pin -dir I -type clk dphy_clk_200M
  create_bd_pin -dir O -type intr iic2intc_irpt
  create_bd_pin -dir I -type clk s_axi_aclk
  create_bd_pin -dir I -type rst s_axi_aresetn
  create_bd_pin -dir I -type clk video_aclk
  create_bd_pin -dir I -type rst video_aresetn

  # Create instance: axi_iic_1_sensor, and set properties
  set axi_iic_1_sensor [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_iic:2.0 axi_iic_1_sensor ]
  set_property -dict [ list \
   CONFIG.IIC_FREQ_KHZ {400} \
 ] $axi_iic_1_sensor

  # Create instance: mipi_csi2_rx_subsyst_0, and set properties
  set mipi_csi2_rx_subsyst_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:mipi_csi2_rx_subsystem:5.1 mipi_csi2_rx_subsyst_0 ]
  set_property -dict [ list \
   CONFIG.CMN_NUM_LANES {4} \
   CONFIG.CMN_NUM_PIXELS {4} \
   CONFIG.CMN_PXL_FORMAT {RAW10} \
   CONFIG.CMN_VC {0} \
   CONFIG.CSI_BUF_DEPTH {4096} \
   CONFIG.C_CSI2RX_DBG {1} \
   CONFIG.C_CSI_EN_ACTIVELANES {true} \
   CONFIG.C_CSI_FILTER_USERDATATYPE {true} \
   CONFIG.C_DPHY_LANES {4} \
   CONFIG.C_HS_LINE_RATE {1440} \
   CONFIG.C_HS_SETTLE_NS {141} \
   CONFIG.C_STRETCH_LINE_RATE {2500} \
   CONFIG.DPY_EN_REG_IF {false} \
   CONFIG.DPY_LINE_RATE {1440} \
   CONFIG.SupportLevel {1} \
 ] $mipi_csi2_rx_subsyst_0

  # Create interface connections
  connect_bd_intf_net -intf_net axi_iic_1_sensor_IIC [get_bd_intf_pins IIC_sensor] [get_bd_intf_pins axi_iic_1_sensor/IIC]
  connect_bd_intf_net -intf_net mipi_csi2_rx_subsyst_0_video_out [get_bd_intf_pins video_out] [get_bd_intf_pins mipi_csi2_rx_subsyst_0/video_out]
  connect_bd_intf_net -intf_net mipi_phy_csi_1 [get_bd_intf_pins mipi_phy_csi] [get_bd_intf_pins mipi_csi2_rx_subsyst_0/mipi_phy_if]
  connect_bd_intf_net -intf_net smartconnect_0_M00_AXI [get_bd_intf_pins csirxss_s_axi] [get_bd_intf_pins mipi_csi2_rx_subsyst_0/csirxss_s_axi]
  connect_bd_intf_net -intf_net smartconnect_0_M02_AXI [get_bd_intf_pins S_AXI] [get_bd_intf_pins axi_iic_1_sensor/S_AXI]

  # Create port connections
  connect_bd_net -net axi_iic_1_sensor_iic2intc_irpt [get_bd_pins iic2intc_irpt] [get_bd_pins axi_iic_1_sensor/iic2intc_irpt]
  connect_bd_net -net clk_wizard_0_clk_out1 [get_bd_pins s_axi_aclk] [get_bd_pins axi_iic_1_sensor/s_axi_aclk] [get_bd_pins mipi_csi2_rx_subsyst_0/lite_aclk]
  connect_bd_net -net clk_wizard_0_clk_out2 [get_bd_pins dphy_clk_200M] [get_bd_pins mipi_csi2_rx_subsyst_0/dphy_clk_200M]
  connect_bd_net -net clk_wizard_0_clk_out3 [get_bd_pins video_aclk] [get_bd_pins mipi_csi2_rx_subsyst_0/video_aclk]
  connect_bd_net -net mipi_csi2_rx_subsyst_0_csirxss_csi_irq [get_bd_pins csirxss_csi_irq] [get_bd_pins mipi_csi2_rx_subsyst_0/csirxss_csi_irq]
  connect_bd_net -net proc_sys_reset_0_peripheral_aresetn [get_bd_pins s_axi_aresetn] [get_bd_pins axi_iic_1_sensor/s_axi_aresetn] [get_bd_pins mipi_csi2_rx_subsyst_0/lite_aresetn]
  connect_bd_net -net proc_sys_reset_1_peripheral_aresetn [get_bd_pins video_aresetn] [get_bd_pins mipi_csi2_rx_subsyst_0/video_aresetn]

  # Restore current instance
  current_bd_instance $oldCurInst
}

# Hierarchical cell: cap_pipe
proc create_hier_cell_cap_pipe { parentCell nameHier } {

  variable script_folder

  if { $parentCell eq "" || $nameHier eq "" } {
     catch {common::send_gid_msg -ssname BD::TCL -id 2092 -severity "ERROR" "create_hier_cell_cap_pipe() - Empty argument(s)!"}
     return
  }

  # Get object for parentCell
  set parentObj [get_bd_cells $parentCell]
  if { $parentObj == "" } {
     catch {common::send_gid_msg -ssname BD::TCL -id 2090 -severity "ERROR" "Unable to find parent cell <$parentCell>!"}
     return
  }

  # Make sure parentObj is hier blk
  set parentType [get_property TYPE $parentObj]
  if { $parentType ne "hier" } {
     catch {common::send_gid_msg -ssname BD::TCL -id 2091 -severity "ERROR" "Parent <$parentObj> has TYPE = <$parentType>. Expected to be <hier>."}
     return
  }

  # Save current instance; Restore later
  set oldCurInst [current_bd_instance .]

  # Set parent object as current
  current_bd_instance $parentObj

  # Create cell and set as current instance
  set hier_obj [create_bd_cell -type hier $nameHier]
  current_bd_instance $hier_obj

  # Create interface pins
  create_bd_intf_pin -mode Master -vlnv xilinx.com:interface:aximm_rtl:1.0 M00_AXI

  create_bd_intf_pin -mode Slave -vlnv xilinx.com:interface:axis_rtl:1.0 S_AXIS

  create_bd_intf_pin -mode Slave -vlnv xilinx.com:interface:aximm_rtl:1.0 s_axi_CTRL

  create_bd_intf_pin -mode Slave -vlnv xilinx.com:interface:aximm_rtl:1.0 s_axi_CTRL1

  create_bd_intf_pin -mode Slave -vlnv xilinx.com:interface:aximm_rtl:1.0 s_axi_ctrl_1


  # Create pins
  create_bd_pin -dir I -from 31 -to 0 Din
  create_bd_pin -dir O -from 0 -to 0 Dout
  create_bd_pin -dir O -from 0 -to 0 dout_1
  create_bd_pin -dir O -type intr frm_buf_irq
  create_bd_pin -dir I -type clk video_clk
  create_bd_pin -dir I -type rst video_rst_n

  # Create instance: ISPPipeline_accel_0, and set properties
  set ISPPipeline_accel_0 [ create_bd_cell -type ip -vlnv xilinx.com:hls:ISPPipeline_accel:1.0 ISPPipeline_accel_0 ]

  # Create instance: axis_data_fifo_cap, and set properties
  set axis_data_fifo_cap [ create_bd_cell -type ip -vlnv xilinx.com:ip:axis_data_fifo:2.0 axis_data_fifo_cap ]
  set_property -dict [ list \
   CONFIG.FIFO_DEPTH {8192} \
 ] $axis_data_fifo_cap

  # Create instance: axis_subset_converter_0, and set properties
  set axis_subset_converter_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:axis_subset_converter:1.1 axis_subset_converter_0 ]
  set_property -dict [ list \
   CONFIG.M_TDATA_NUM_BYTES {5} \
   CONFIG.S_TDATA_NUM_BYTES {5} \
   CONFIG.TDATA_REMAP {tdata[39:0]} \
 ] $axis_subset_converter_0

  # Create instance: v_frmbuf_wr_0, and set properties
  set v_frmbuf_wr_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:v_frmbuf_wr:2.2 v_frmbuf_wr_0 ]
  set_property -dict [ list \
   CONFIG.AXIMM_ADDR_WIDTH {64} \
   CONFIG.AXIMM_DATA_WIDTH {256} \
   CONFIG.C_M_AXI_MM_VIDEO_DATA_WIDTH {256} \
   CONFIG.HAS_BGR8 {0} \
   CONFIG.HAS_BGRX8 {0} \
   CONFIG.HAS_RGBX8 {0} \
   CONFIG.HAS_UYVY8 {0} \
   CONFIG.HAS_Y8 {0} \
   CONFIG.HAS_YUV8 {0} \
   CONFIG.HAS_YUVX8 {0} \
   CONFIG.HAS_YUYV8 {1} \
   CONFIG.HAS_Y_UV8 {0} \
   CONFIG.MAX_NR_PLANES {1} \
   CONFIG.SAMPLES_PER_CLOCK {4} \
 ] $v_frmbuf_wr_0

  # Create instance: v_proc_ss_0, and set properties
  set v_proc_ss_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:v_proc_ss:2.3 v_proc_ss_0 ]
  set_property -dict [ list \
   CONFIG.C_COLORSPACE_SUPPORT {1} \
   CONFIG.C_ENABLE_CSC {true} \
   CONFIG.C_MAX_DATA_WIDTH {8} \
   CONFIG.C_SAMPLES_PER_CLK {4} \
   CONFIG.C_TOPOLOGY {0} \
 ] $v_proc_ss_0

  # Create instance: vcc, and set properties
  set vcc [ create_bd_cell -type ip -vlnv xilinx.com:ip:xlconstant:1.1 vcc ]

  # Create instance: xlslice_0, and set properties
  set xlslice_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:xlslice:1.0 xlslice_0 ]
  set_property -dict [ list \
   CONFIG.DIN_FROM {1} \
   CONFIG.DIN_TO {1} \
   CONFIG.DOUT_WIDTH {1} \
 ] $xlslice_0

  # Create instance: xlslice_1, and set properties
  set xlslice_1 [ create_bd_cell -type ip -vlnv xilinx.com:ip:xlslice:1.0 xlslice_1 ]
  set_property -dict [ list \
   CONFIG.DIN_FROM {2} \
   CONFIG.DIN_TO {2} \
   CONFIG.DOUT_WIDTH {1} \
 ] $xlslice_1

  # Create instance: xlslice_2, and set properties
  set xlslice_2 [ create_bd_cell -type ip -vlnv xilinx.com:ip:xlslice:1.0 xlslice_2 ]
  set_property -dict [ list \
   CONFIG.DIN_FROM {3} \
   CONFIG.DIN_TO {3} \
   CONFIG.DOUT_WIDTH {1} \
 ] $xlslice_2

  # Create instance: xlslice_3, and set properties
  set xlslice_3 [ create_bd_cell -type ip -vlnv xilinx.com:ip:xlslice:1.0 xlslice_3 ]
  set_property -dict [ list \
   CONFIG.DIN_FROM {4} \
   CONFIG.DIN_TO {4} \
   CONFIG.DOUT_WIDTH {1} \
 ] $xlslice_3

  # Create interface connections
  connect_bd_intf_net -intf_net ISPPipeline_accel_0_m_axis_video [get_bd_intf_pins ISPPipeline_accel_0/m_axis_video] [get_bd_intf_pins v_proc_ss_0/s_axis]
  connect_bd_intf_net -intf_net axis_data_fifo_0_M_AXIS [get_bd_intf_pins ISPPipeline_accel_0/s_axis_video] [get_bd_intf_pins axis_data_fifo_cap/M_AXIS]
  connect_bd_intf_net -intf_net axis_subset_converter_0_M_AXIS [get_bd_intf_pins axis_data_fifo_cap/S_AXIS] [get_bd_intf_pins axis_subset_converter_0/M_AXIS]
  connect_bd_intf_net -intf_net mipi_csi2_rx_subsyst_0_video_out [get_bd_intf_pins S_AXIS] [get_bd_intf_pins axis_subset_converter_0/S_AXIS]
  connect_bd_intf_net -intf_net s_axi_ctrl_1_1 [get_bd_intf_pins s_axi_ctrl_1] [get_bd_intf_pins v_proc_ss_0/s_axi_ctrl]
  connect_bd_intf_net -intf_net smartconnect_2_M00_AXI [get_bd_intf_pins s_axi_CTRL1] [get_bd_intf_pins ISPPipeline_accel_0/s_axi_CTRL]
  connect_bd_intf_net -intf_net smartconnect_2_M03_AXI [get_bd_intf_pins s_axi_CTRL] [get_bd_intf_pins v_frmbuf_wr_0/s_axi_CTRL]
  connect_bd_intf_net -intf_net v_frmbuf_wr_0_m_axi_mm_video [get_bd_intf_pins M00_AXI] [get_bd_intf_pins v_frmbuf_wr_0/m_axi_mm_video]
  connect_bd_intf_net -intf_net v_proc_ss_0_m_axis [get_bd_intf_pins v_frmbuf_wr_0/s_axis_video] [get_bd_intf_pins v_proc_ss_0/m_axis]

  # Create port connections
  connect_bd_net -net ap_rst_n_1 [get_bd_pins video_rst_n] [get_bd_pins axis_data_fifo_cap/s_axis_aresetn] [get_bd_pins axis_subset_converter_0/aresetn]
  connect_bd_net -net axi_gpio_0_gpio_io_o [get_bd_pins Din] [get_bd_pins xlslice_0/Din] [get_bd_pins xlslice_1/Din] [get_bd_pins xlslice_2/Din] [get_bd_pins xlslice_3/Din]
  connect_bd_net -net clk_wizard_0_clk_out3 [get_bd_pins video_clk] [get_bd_pins ISPPipeline_accel_0/ap_clk] [get_bd_pins axis_data_fifo_cap/s_axis_aclk] [get_bd_pins axis_subset_converter_0/aclk] [get_bd_pins v_frmbuf_wr_0/ap_clk] [get_bd_pins v_proc_ss_0/aclk_axis] [get_bd_pins v_proc_ss_0/aclk_ctrl]
  connect_bd_net -net v_frmbuf_wr_0_interrupt [get_bd_pins frm_buf_irq] [get_bd_pins v_frmbuf_wr_0/interrupt]
  connect_bd_net -net vcc_dout [get_bd_pins dout_1] [get_bd_pins vcc/dout]
  connect_bd_net -net xlslice_0_Dout [get_bd_pins ISPPipeline_accel_0/ap_rst_n] [get_bd_pins xlslice_0/Dout]
  connect_bd_net -net xlslice_3_Dout [get_bd_pins v_proc_ss_0/aresetn_ctrl] [get_bd_pins xlslice_1/Dout]
  connect_bd_net -net xlslice_4_Dout [get_bd_pins v_frmbuf_wr_0/ap_rst_n] [get_bd_pins xlslice_2/Dout]
  connect_bd_net -net xlslice_5_Dout [get_bd_pins Dout] [get_bd_pins xlslice_3/Dout]

  # Restore current instance
  current_bd_instance $oldCurInst
}

# Hierarchical cell: hdmi_tx_phy
proc create_hier_cell_hdmi_tx_phy { parentCell nameHier } {

  variable script_folder

  if { $parentCell eq "" || $nameHier eq "" } {
     catch {common::send_gid_msg -ssname BD::TCL -id 2092 -severity "ERROR" "create_hier_cell_hdmi_tx_phy() - Empty argument(s)!"}
     return
  }

  # Get object for parentCell
  set parentObj [get_bd_cells $parentCell]
  if { $parentObj == "" } {
     catch {common::send_gid_msg -ssname BD::TCL -id 2090 -severity "ERROR" "Unable to find parent cell <$parentCell>!"}
     return
  }

  # Make sure parentObj is hier blk
  set parentType [get_property TYPE $parentObj]
  if { $parentType ne "hier" } {
     catch {common::send_gid_msg -ssname BD::TCL -id 2091 -severity "ERROR" "Parent <$parentObj> has TYPE = <$parentType>. Expected to be <hier>."}
     return
  }

  # Save current instance; Restore later
  set oldCurInst [current_bd_instance .]

  # Set parent object as current
  current_bd_instance $parentObj

  # Create cell and set as current instance
  set hier_obj [create_bd_cell -type hier $nameHier]
  current_bd_instance $hier_obj

  # Create interface pins
  create_bd_intf_pin -mode Master -vlnv xilinx.com:interface:iic_rtl:1.0 HDMI_CTL_IIC

  create_bd_intf_pin -mode Master -vlnv xilinx.com:interface:axis_rtl:1.0 M_AXIS

  create_bd_intf_pin -mode Slave -vlnv xilinx.com:interface:aximm_rtl:1.0 S_AXI

  create_bd_intf_pin -mode Slave -vlnv xilinx.com:interface:aximm_rtl:1.0 axi4lite

  create_bd_intf_pin -mode Slave -vlnv xilinx.com:interface:axis_rtl:1.0 s_axis_video

  create_bd_intf_pin -mode Master -vlnv xilinx.com:interface:axis_rtl:1.0 status_sb_tx

  create_bd_intf_pin -mode Slave -vlnv xilinx.com:interface:axis_rtl:1.0 tx_axi4s_ch0

  create_bd_intf_pin -mode Slave -vlnv xilinx.com:interface:axis_rtl:1.0 tx_axi4s_ch1

  create_bd_intf_pin -mode Slave -vlnv xilinx.com:interface:axis_rtl:1.0 tx_axi4s_ch2


  # Create pins
  create_bd_pin -dir I -type rst ARESETN
  create_bd_pin -dir I IDT_8T49N241_LOL_IN
  create_bd_pin -dir I -from 3 -to 0 RX_DATA_IN_rxn
  create_bd_pin -dir I -from 3 -to 0 RX_DATA_IN_rxp
  create_bd_pin -dir O -from 3 -to 0 TX_DATA_OUT_txn
  create_bd_pin -dir O -from 3 -to 0 TX_DATA_OUT_txp
  create_bd_pin -dir O -from 0 -to 0 -type rst TX_EN_OUT
  create_bd_pin -dir I -from 0 -to 0 -type clk TX_REFCLK_N_IN
  create_bd_pin -dir I -from 0 -to 0 -type clk TX_REFCLK_P_IN
  create_bd_pin -dir I -type clk aclk
  create_bd_pin -dir I altclk
  create_bd_pin -dir I -type rst aresetn1
  create_bd_pin -dir O -type intr iic2intc_irpt
  create_bd_pin -dir O -type intr irq
  create_bd_pin -dir O -type clk tx_tmds_clk
  create_bd_pin -dir O -type gt_usrclk tx_usrclk
  create_bd_pin -dir O -type clk tx_video_clk

  # Create instance: GT_Quad_and_Clk
  create_hier_cell_GT_Quad_and_Clk $hier_obj GT_Quad_and_Clk

  # Create instance: fmch_axi_iic, and set properties
  set fmch_axi_iic [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_iic:2.0 fmch_axi_iic ]

  # Create instance: gt_refclk1
  create_hier_cell_gt_refclk1 $hier_obj gt_refclk1

  # Create instance: hdmi_gt_controller_1, and set properties
  set hdmi_gt_controller_1 [ create_bd_cell -type ip -vlnv xilinx.com:ip:hdmi_gt_controller:1.0 hdmi_gt_controller_1 ]
  set_property -dict [ list \
   CONFIG.C_GT_DEBUG_PORT_EN {true} \
   CONFIG.C_GT_DIRECTION {SIMPLEX_TX} \
   CONFIG.C_INPUT_PIXELS_PER_CLOCK {4} \
   CONFIG.C_NIDRU {false} \
   CONFIG.C_NIDRU_REFCLK_SEL {2} \
   CONFIG.C_RX_PLL_SELECTION {8} \
   CONFIG.C_RX_REFCLK_SEL {0} \
   CONFIG.C_Rx_Protocol {None} \
   CONFIG.C_TX_PLL_SELECTION {7} \
   CONFIG.C_TX_REFCLK_SEL {1} \
   CONFIG.C_Tx_No_Of_Channels {4} \
   CONFIG.C_Tx_Protocol {HDMI} \
   CONFIG.C_Txrefclk_Rdy_Invert {true} \
   CONFIG.C_Use_GT_CH4_HDMI {true} \
   CONFIG.C_Use_Oddr_for_Tmds_Clkout {false} \
   CONFIG.C_vid_phy_rx_axi4s_ch_INT_TDATA_WIDTH {40} \
   CONFIG.C_vid_phy_rx_axi4s_ch_TDATA_WIDTH {40} \
   CONFIG.C_vid_phy_tx_axi4s_ch_INT_TDATA_WIDTH {40} \
   CONFIG.C_vid_phy_tx_axi4s_ch_TDATA_WIDTH {40} \
   CONFIG.Transceiver_Width {4} \
   CONFIG.check_refclk_selection {0} \
 ] $hdmi_gt_controller_1

  # Create instance: tx_video_axis_reg_slice, and set properties
  set tx_video_axis_reg_slice [ create_bd_cell -type ip -vlnv xilinx.com:ip:axis_register_slice:1.1 tx_video_axis_reg_slice ]

  # Create instance: vcc_const, and set properties
  set vcc_const [ create_bd_cell -type ip -vlnv xilinx.com:ip:xlconstant:1.1 vcc_const ]
  set_property -dict [ list \
   CONFIG.CONST_VAL {1} \
 ] $vcc_const

  # Create interface connections
  connect_bd_intf_net -intf_net Conn1 [get_bd_intf_pins M_AXIS] [get_bd_intf_pins tx_video_axis_reg_slice/M_AXIS]
  connect_bd_intf_net -intf_net Conn2 [get_bd_intf_pins tx_axi4s_ch0] [get_bd_intf_pins hdmi_gt_controller_1/tx_axi4s_ch0]
  connect_bd_intf_net -intf_net Conn3 [get_bd_intf_pins tx_axi4s_ch1] [get_bd_intf_pins hdmi_gt_controller_1/tx_axi4s_ch1]
  connect_bd_intf_net -intf_net Conn4 [get_bd_intf_pins tx_axi4s_ch2] [get_bd_intf_pins hdmi_gt_controller_1/tx_axi4s_ch2]
  connect_bd_intf_net -intf_net Conn5 [get_bd_intf_pins status_sb_tx] [get_bd_intf_pins hdmi_gt_controller_1/status_sb_tx]
  connect_bd_intf_net -intf_net hdmi_gt_controller_1_ch0_debug [get_bd_intf_pins GT_Quad_and_Clk/CH0_DEBUG] [get_bd_intf_pins hdmi_gt_controller_1/ch0_debug]
  connect_bd_intf_net -intf_net hdmi_gt_controller_1_ch1_debug [get_bd_intf_pins GT_Quad_and_Clk/CH1_DEBUG] [get_bd_intf_pins hdmi_gt_controller_1/ch1_debug]
  connect_bd_intf_net -intf_net hdmi_gt_controller_1_ch2_debug [get_bd_intf_pins GT_Quad_and_Clk/CH2_DEBUG] [get_bd_intf_pins hdmi_gt_controller_1/ch2_debug]
  connect_bd_intf_net -intf_net hdmi_gt_controller_1_ch3_debug [get_bd_intf_pins GT_Quad_and_Clk/CH3_DEBUG] [get_bd_intf_pins hdmi_gt_controller_1/ch3_debug]
  connect_bd_intf_net -intf_net hdmi_gt_controller_1_gt_debug [get_bd_intf_pins GT_Quad_and_Clk/GT_DEBUG] [get_bd_intf_pins hdmi_gt_controller_1/gt_debug]
  connect_bd_intf_net -intf_net hdmi_gt_controller_1_gt_tx0 [get_bd_intf_pins GT_Quad_and_Clk/TX0_GT_IP_Interface] [get_bd_intf_pins hdmi_gt_controller_1/gt_tx0]
  connect_bd_intf_net -intf_net hdmi_gt_controller_1_gt_tx1 [get_bd_intf_pins GT_Quad_and_Clk/TX1_GT_IP_Interface] [get_bd_intf_pins hdmi_gt_controller_1/gt_tx1]
  connect_bd_intf_net -intf_net hdmi_gt_controller_1_gt_tx2 [get_bd_intf_pins GT_Quad_and_Clk/TX2_GT_IP_Interface] [get_bd_intf_pins hdmi_gt_controller_1/gt_tx2]
  connect_bd_intf_net -intf_net hdmi_gt_controller_1_gt_tx3 [get_bd_intf_pins GT_Quad_and_Clk/TX3_GT_IP_Interface] [get_bd_intf_pins hdmi_gt_controller_1/gt_tx3]
  connect_bd_intf_net -intf_net hdmi_tx_pipe_HDMI_CTL_IIC [get_bd_intf_pins HDMI_CTL_IIC] [get_bd_intf_pins fmch_axi_iic/IIC]
  connect_bd_intf_net -intf_net s_axis_video_1 [get_bd_intf_pins s_axis_video] [get_bd_intf_pins tx_video_axis_reg_slice/S_AXIS]
  connect_bd_intf_net -intf_net smartconnect_100mhz_M00_AXI [get_bd_intf_pins axi4lite] [get_bd_intf_pins hdmi_gt_controller_1/axi4lite]
  connect_bd_intf_net -intf_net smartconnect_100mhz_M04_AXI [get_bd_intf_pins S_AXI] [get_bd_intf_pins fmch_axi_iic/S_AXI]

  # Create port connections
  connect_bd_net -net GT_Quad_and_Clk_ch0_iloresetdone [get_bd_pins GT_Quad_and_Clk/ch0_iloresetdone] [get_bd_pins hdmi_gt_controller_1/gt_ch0_ilo_resetdone]
  connect_bd_net -net GT_Quad_and_Clk_ch1_iloresetdone [get_bd_pins GT_Quad_and_Clk/ch1_iloresetdone] [get_bd_pins hdmi_gt_controller_1/gt_ch1_ilo_resetdone]
  connect_bd_net -net GT_Quad_and_Clk_ch2_iloresetdone [get_bd_pins GT_Quad_and_Clk/ch2_iloresetdone] [get_bd_pins hdmi_gt_controller_1/gt_ch2_ilo_resetdone]
  connect_bd_net -net GT_Quad_and_Clk_ch3_iloresetdone [get_bd_pins GT_Quad_and_Clk/ch3_iloresetdone] [get_bd_pins hdmi_gt_controller_1/gt_ch3_ilo_resetdone]
  connect_bd_net -net GT_Quad_and_Clk_gtpowergood [get_bd_pins GT_Quad_and_Clk/gtpowergood] [get_bd_pins hdmi_gt_controller_1/gtpowergood]
  connect_bd_net -net GT_Quad_and_Clk_hsclk0_lcplllock [get_bd_pins GT_Quad_and_Clk/hsclk0_lcplllock] [get_bd_pins hdmi_gt_controller_1/gt_lcpll0_lock]
  connect_bd_net -net GT_Quad_and_Clk_hsclk1_lcplllock [get_bd_pins GT_Quad_and_Clk/hsclk1_lcplllock] [get_bd_pins hdmi_gt_controller_1/gt_lcpll1_lock]
  connect_bd_net -net GT_Quad_and_Clk_txn_0 [get_bd_pins TX_DATA_OUT_txn] [get_bd_pins GT_Quad_and_Clk/TX_DATA_OUT_txn]
  connect_bd_net -net GT_Quad_and_Clk_txp_0 [get_bd_pins TX_DATA_OUT_txp] [get_bd_pins GT_Quad_and_Clk/TX_DATA_OUT_txp]
  connect_bd_net -net RX_DATA_IN_rxn [get_bd_pins RX_DATA_IN_rxn] [get_bd_pins GT_Quad_and_Clk/RX_DATA_IN_rxn]
  connect_bd_net -net RX_DATA_IN_rxp [get_bd_pins RX_DATA_IN_rxp] [get_bd_pins GT_Quad_and_Clk/RX_DATA_IN_rxp]
  connect_bd_net -net TX_REFCLK_N_IN_1 [get_bd_pins TX_REFCLK_N_IN] [get_bd_pins gt_refclk1/CLK_N_IN]
  connect_bd_net -net TX_REFCLK_P_IN_1 [get_bd_pins TX_REFCLK_P_IN] [get_bd_pins gt_refclk1/CLK_P_IN]
  connect_bd_net -net bufg_gt_1_usrclk [get_bd_pins tx_usrclk] [get_bd_pins GT_Quad_and_Clk/tx_usrclk] [get_bd_pins hdmi_gt_controller_1/gt_txusrclk] [get_bd_pins hdmi_gt_controller_1/tx_axi4s_aclk]
  connect_bd_net -net fmch_axi_iic_iic2intc_irpt [get_bd_pins iic2intc_irpt] [get_bd_pins fmch_axi_iic/iic2intc_irpt]
  connect_bd_net -net gt_refclk1_O [get_bd_pins GT_Quad_and_Clk/GT_REFCLK1] [get_bd_pins gt_refclk1/O]
  connect_bd_net -net gt_refclk1_ODIV2 [get_bd_pins gt_refclk1/ODIV2] [get_bd_pins hdmi_gt_controller_1/gt_refclk1_odiv2]
  connect_bd_net -net hdmi_gt_controller_1_gt_lcpll0_reset [get_bd_pins GT_Quad_and_Clk/hsclk0_lcpllreset] [get_bd_pins hdmi_gt_controller_1/gt_lcpll0_reset]
  connect_bd_net -net hdmi_gt_controller_1_gt_lcpll1_reset [get_bd_pins GT_Quad_and_Clk/hsclk1_lcpllreset] [get_bd_pins hdmi_gt_controller_1/gt_lcpll1_reset]
  connect_bd_net -net hdmi_gt_controller_1_irq [get_bd_pins irq] [get_bd_pins hdmi_gt_controller_1/irq]
  connect_bd_net -net hdmi_gt_controller_1_tx_tmds_clk [get_bd_pins tx_tmds_clk] [get_bd_pins hdmi_gt_controller_1/tx_tmds_clk]
  connect_bd_net -net hdmi_gt_controller_1_tx_video_clk [get_bd_pins tx_video_clk] [get_bd_pins hdmi_gt_controller_1/tx_video_clk]
  connect_bd_net -net net_bdry_in_SI5324_LOL_IN [get_bd_pins IDT_8T49N241_LOL_IN] [get_bd_pins hdmi_gt_controller_1/tx_refclk_rdy]
  connect_bd_net -net net_mb_ss_0_clk_out2 [get_bd_pins aclk] [get_bd_pins tx_video_axis_reg_slice/aclk]
  connect_bd_net -net net_mb_ss_0_dcm_locked [get_bd_pins aresetn1] [get_bd_pins tx_video_axis_reg_slice/aresetn]
  connect_bd_net -net net_mb_ss_0_peripheral_aresetn [get_bd_pins ARESETN] [get_bd_pins fmch_axi_iic/s_axi_aresetn] [get_bd_pins hdmi_gt_controller_1/axi4lite_aresetn] [get_bd_pins hdmi_gt_controller_1/sb_aresetn]
  connect_bd_net -net net_mb_ss_0_s_axi_aclk [get_bd_pins altclk] [get_bd_pins GT_Quad_and_Clk/altclk] [get_bd_pins fmch_axi_iic/s_axi_aclk] [get_bd_pins hdmi_gt_controller_1/apb_clk] [get_bd_pins hdmi_gt_controller_1/axi4lite_aclk] [get_bd_pins hdmi_gt_controller_1/sb_aclk]
  connect_bd_net -net net_vcc_const_dout [get_bd_pins TX_EN_OUT] [get_bd_pins hdmi_gt_controller_1/tx_axi4s_aresetn] [get_bd_pins vcc_const/dout]

  # Restore current instance
  current_bd_instance $oldCurInst
}

# Hierarchical cell: refclk_aud
proc create_hier_cell_refclk_aud { parentCell nameHier } {

  variable script_folder

  if { $parentCell eq "" || $nameHier eq "" } {
     catch {common::send_gid_msg -ssname BD::TCL -id 2092 -severity "ERROR" "create_hier_cell_refclk_aud() - Empty argument(s)!"}
     return
  }

  # Get object for parentCell
  set parentObj [get_bd_cells $parentCell]
  if { $parentObj == "" } {
     catch {common::send_gid_msg -ssname BD::TCL -id 2090 -severity "ERROR" "Unable to find parent cell <$parentCell>!"}
     return
  }

  # Make sure parentObj is hier blk
  set parentType [get_property TYPE $parentObj]
  if { $parentType ne "hier" } {
     catch {common::send_gid_msg -ssname BD::TCL -id 2091 -severity "ERROR" "Parent <$parentObj> has TYPE = <$parentType>. Expected to be <hier>."}
     return
  }

  # Save current instance; Restore later
  set oldCurInst [current_bd_instance .]

  # Set parent object as current
  current_bd_instance $parentObj

  # Create cell and set as current instance
  set hier_obj [create_bd_cell -type hier $nameHier]
  current_bd_instance $hier_obj

  # Create interface pins
  create_bd_intf_pin -mode Slave -vlnv xilinx.com:interface:diff_clock_rtl:1.0 CLK_IN_D_0


  # Create pins
  create_bd_pin -dir O -from 0 -to 0 -type clk BUFG_O

  # Create instance: util_ds_buf_0, and set properties
  set util_ds_buf_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:util_ds_buf:2.1 util_ds_buf_0 ]

  # Create instance: util_ds_buf_1, and set properties
  set util_ds_buf_1 [ create_bd_cell -type ip -vlnv xilinx.com:ip:util_ds_buf:2.1 util_ds_buf_1 ]
  set_property -dict [ list \
   CONFIG.C_BUF_TYPE {BUFG} \
 ] $util_ds_buf_1

  # Create interface connections
  connect_bd_intf_net -intf_net Conn3 [get_bd_intf_pins CLK_IN_D_0] [get_bd_intf_pins util_ds_buf_0/CLK_IN_D]

  # Create port connections
  connect_bd_net -net util_ds_buf_0_IBUF_OUT [get_bd_pins util_ds_buf_0/IBUF_OUT] [get_bd_pins util_ds_buf_1/BUFG_I]
  connect_bd_net -net util_ds_buf_1_BUFG_O [get_bd_pins BUFG_O] [get_bd_pins util_ds_buf_1/BUFG_O]

  # Restore current instance
  current_bd_instance $oldCurInst
}

# Hierarchical cell: mipi_capture_pipe
proc create_hier_cell_mipi_capture_pipe { parentCell nameHier } {

  variable script_folder

  if { $parentCell eq "" || $nameHier eq "" } {
     catch {common::send_gid_msg -ssname BD::TCL -id 2092 -severity "ERROR" "create_hier_cell_mipi_capture_pipe() - Empty argument(s)!"}
     return
  }

  # Get object for parentCell
  set parentObj [get_bd_cells $parentCell]
  if { $parentObj == "" } {
     catch {common::send_gid_msg -ssname BD::TCL -id 2090 -severity "ERROR" "Unable to find parent cell <$parentCell>!"}
     return
  }

  # Make sure parentObj is hier blk
  set parentType [get_property TYPE $parentObj]
  if { $parentType ne "hier" } {
     catch {common::send_gid_msg -ssname BD::TCL -id 2091 -severity "ERROR" "Parent <$parentObj> has TYPE = <$parentType>. Expected to be <hier>."}
     return
  }

  # Save current instance; Restore later
  set oldCurInst [current_bd_instance .]

  # Set parent object as current
  current_bd_instance $parentObj

  # Create cell and set as current instance
  set hier_obj [create_bd_cell -type hier $nameHier]
  current_bd_instance $hier_obj

  # Create interface pins
  create_bd_intf_pin -mode Master -vlnv xilinx.com:interface:aximm_rtl:1.0 M00_AXI

  create_bd_intf_pin -mode Slave -vlnv xilinx.com:interface:aximm_rtl:1.0 S_AXI

  create_bd_intf_pin -mode Slave -vlnv xilinx.com:interface:mipi_phy_rtl:1.0 csi_mipi_phy_if

  create_bd_intf_pin -mode Slave -vlnv xilinx.com:interface:aximm_rtl:1.0 csirxss_s_axi

  create_bd_intf_pin -mode Slave -vlnv xilinx.com:interface:aximm_rtl:1.0 s_axi_CTRL

  create_bd_intf_pin -mode Slave -vlnv xilinx.com:interface:aximm_rtl:1.0 s_axi_CTRL1

  create_bd_intf_pin -mode Slave -vlnv xilinx.com:interface:aximm_rtl:1.0 s_axi_ctrl_1

  create_bd_intf_pin -mode Master -vlnv xilinx.com:interface:iic_rtl:1.0 sensor_iic


  # Create pins
  create_bd_pin -dir I -from 31 -to 0 Din
  create_bd_pin -dir O -type intr csirxss_csi_irq
  create_bd_pin -dir O -type intr frm_buf_irq
  create_bd_pin -dir O -type intr iic2intc_irpt
  create_bd_pin -dir I -type clk s_axi_aclk
  create_bd_pin -dir I -type rst s_axi_aresetn
  create_bd_pin -dir O -from 0 -to 0 sensor_gpio_rst
  create_bd_pin -dir O -from 0 -to 0 sensor_gpio_spi_cs_n
  create_bd_pin -dir I -type clk video_clk
  create_bd_pin -dir I -type rst video_rst_n

  # Create instance: cap_pipe
  create_hier_cell_cap_pipe $hier_obj cap_pipe

  # Create instance: mipi_csi_rx_ss
  create_hier_cell_mipi_csi_rx_ss $hier_obj mipi_csi_rx_ss

  # Create interface connections
  connect_bd_intf_net -intf_net cap_pipe_M00_AXI [get_bd_intf_pins M00_AXI] [get_bd_intf_pins cap_pipe/M00_AXI]
  connect_bd_intf_net -intf_net csi_mipi_phy_if_1 [get_bd_intf_pins csi_mipi_phy_if] [get_bd_intf_pins mipi_csi_rx_ss/mipi_phy_csi]
  connect_bd_intf_net -intf_net csirxss_s_axi_1 [get_bd_intf_pins csirxss_s_axi] [get_bd_intf_pins mipi_csi_rx_ss/csirxss_s_axi]
  connect_bd_intf_net -intf_net mipi_csi_rx_ss_IIC_sensor [get_bd_intf_pins sensor_iic] [get_bd_intf_pins mipi_csi_rx_ss/IIC_sensor]
  connect_bd_intf_net -intf_net mipi_csi_rx_ss_video_out [get_bd_intf_pins cap_pipe/S_AXIS] [get_bd_intf_pins mipi_csi_rx_ss/video_out]
  connect_bd_intf_net -intf_net s_axi_CTRL1_1 [get_bd_intf_pins s_axi_CTRL1] [get_bd_intf_pins cap_pipe/s_axi_CTRL1]
  connect_bd_intf_net -intf_net s_axi_CTRL_1 [get_bd_intf_pins s_axi_CTRL] [get_bd_intf_pins cap_pipe/s_axi_CTRL]
  connect_bd_intf_net -intf_net s_axi_ctrl_1_1 [get_bd_intf_pins s_axi_ctrl_1] [get_bd_intf_pins cap_pipe/s_axi_ctrl_1]
  connect_bd_intf_net -intf_net smartconnect_gp0_M04_AXI [get_bd_intf_pins S_AXI] [get_bd_intf_pins mipi_csi_rx_ss/S_AXI]

  # Create port connections
  connect_bd_net -net Din_1 [get_bd_pins Din] [get_bd_pins cap_pipe/Din]
  connect_bd_net -net cap_pipe_Dout [get_bd_pins sensor_gpio_rst] [get_bd_pins cap_pipe/Dout]
  connect_bd_net -net cap_pipe_frm_buf_irq [get_bd_pins frm_buf_irq] [get_bd_pins cap_pipe/frm_buf_irq]
  connect_bd_net -net clk_wiz_clk_out2 [get_bd_pins s_axi_aclk] [get_bd_pins mipi_csi_rx_ss/s_axi_aclk]
  connect_bd_net -net clk_wiz_clk_out3 [get_bd_pins video_clk] [get_bd_pins cap_pipe/video_clk] [get_bd_pins mipi_csi_rx_ss/dphy_clk_200M] [get_bd_pins mipi_csi_rx_ss/video_aclk]
  connect_bd_net -net mipi_csi2_rx_dout1 [get_bd_pins sensor_gpio_spi_cs_n] [get_bd_pins cap_pipe/dout_1]
  connect_bd_net -net mipi_csi_rx_ss_csirxss_csi_irq [get_bd_pins csirxss_csi_irq] [get_bd_pins mipi_csi_rx_ss/csirxss_csi_irq]
  connect_bd_net -net mipi_csi_rx_ss_iic2intc_irpt [get_bd_pins iic2intc_irpt] [get_bd_pins mipi_csi_rx_ss/iic2intc_irpt]
  connect_bd_net -net rst_processor_1_100M_peripheral_aresetn [get_bd_pins s_axi_aresetn] [get_bd_pins mipi_csi_rx_ss/s_axi_aresetn]
  connect_bd_net -net rst_processor_pl_200Mhz_peripheral_aresetn [get_bd_pins video_rst_n] [get_bd_pins cap_pipe/video_rst_n] [get_bd_pins mipi_csi_rx_ss/video_aresetn]

  # Restore current instance
  current_bd_instance $oldCurInst
}

# Hierarchical cell: display_pipe
proc create_hier_cell_display_pipe { parentCell nameHier } {

  variable script_folder

  if { $parentCell eq "" || $nameHier eq "" } {
     catch {common::send_gid_msg -ssname BD::TCL -id 2092 -severity "ERROR" "create_hier_cell_display_pipe() - Empty argument(s)!"}
     return
  }

  # Get object for parentCell
  set parentObj [get_bd_cells $parentCell]
  if { $parentObj == "" } {
     catch {common::send_gid_msg -ssname BD::TCL -id 2090 -severity "ERROR" "Unable to find parent cell <$parentCell>!"}
     return
  }

  # Make sure parentObj is hier blk
  set parentType [get_property TYPE $parentObj]
  if { $parentType ne "hier" } {
     catch {common::send_gid_msg -ssname BD::TCL -id 2091 -severity "ERROR" "Parent <$parentObj> has TYPE = <$parentType>. Expected to be <hier>."}
     return
  }

  # Save current instance; Restore later
  set oldCurInst [current_bd_instance .]

  # Set parent object as current
  current_bd_instance $parentObj

  # Create cell and set as current instance
  set hier_obj [create_bd_cell -type hier $nameHier]
  current_bd_instance $hier_obj

  # Create interface pins
  create_bd_intf_pin -mode Slave -vlnv xilinx.com:interface:axis_rtl:1.0 AUDIO_IN

  create_bd_intf_pin -mode Master -vlnv xilinx.com:interface:iic_rtl:1.0 HDMI_CTL_IIC

  create_bd_intf_pin -mode Slave -vlnv xilinx.com:interface:aximm_rtl:1.0 S_AXI

  create_bd_intf_pin -mode Slave -vlnv xilinx.com:interface:aximm_rtl:1.0 S_AXI_CPU_IN

  create_bd_intf_pin -mode Master -vlnv xilinx.com:interface:iic_rtl:1.0 TX_DDC_OUT

  create_bd_intf_pin -mode Slave -vlnv xilinx.com:interface:aximm_rtl:1.0 axi4lite

  create_bd_intf_pin -mode Slave -vlnv xilinx.com:interface:aximm_rtl:1.0 s_axi_ctrl_vmix

  create_bd_intf_pin -mode Master -vlnv xilinx.com:interface:aximm_rtl:1.0 vmix_mm_axi_vid_rd_0

  create_bd_intf_pin -mode Master -vlnv xilinx.com:interface:aximm_rtl:1.0 vmix_mm_axi_vid_rd_1

  create_bd_intf_pin -mode Master -vlnv xilinx.com:interface:aximm_rtl:1.0 vmix_mm_axi_vid_rd_2


  # Create pins
  create_bd_pin -dir I -type rst ARESETN
  create_bd_pin -dir I -from 31 -to 0 Din
  create_bd_pin -dir I IDT_8T49N241_LOL_IN
  create_bd_pin -dir I -from 3 -to 0 RX_DATA_IN_rxn
  create_bd_pin -dir I -from 3 -to 0 RX_DATA_IN_rxp
  create_bd_pin -dir O -from 3 -to 0 TX_DATA_OUT_txn
  create_bd_pin -dir O -from 3 -to 0 TX_DATA_OUT_txp
  create_bd_pin -dir O -from 0 -to 0 -type rst TX_EN_OUT
  create_bd_pin -dir I TX_HPD_IN
  create_bd_pin -dir I -from 0 -to 0 -type clk TX_REFCLK_N_IN
  create_bd_pin -dir I -from 0 -to 0 -type clk TX_REFCLK_P_IN
  create_bd_pin -dir I -from 19 -to 0 acr_cts
  create_bd_pin -dir I -from 19 -to 0 acr_n
  create_bd_pin -dir I acr_valid
  create_bd_pin -dir I altclk
  create_bd_pin -dir I -type rst aresetn1
  create_bd_pin -dir O -type intr iic2intc_irpt
  create_bd_pin -dir O -type intr irq
  create_bd_pin -dir O -type intr irq1
  create_bd_pin -dir I -type clk s_axis_aclk
  create_bd_pin -dir I -type clk s_axis_audio_aclk
  create_bd_pin -dir I -type rst s_axis_audio_aresetn
  create_bd_pin -dir I -type rst sc_aresetn
  create_bd_pin -dir O -type clk tx_tmds_clk
  create_bd_pin -dir O -type intr vmix_intr

  # Create instance: hdmi_tx_phy
  create_hier_cell_hdmi_tx_phy $hier_obj hdmi_tx_phy

  # Create instance: smartconnect_vmix_0, and set properties
  set smartconnect_vmix_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect:1.0 smartconnect_vmix_0 ]
  set_property -dict [ list \
   CONFIG.ADVANCED_PROPERTIES {    __view__ { functional { S07_Buffer { R_SIZE 1024 } S00_Buffer { R_SIZE 1024 } S06_Buffer { R_SIZE 1024 } S05_Buffer { R_SIZE 1024 } S01_Buffer { R_SIZE 1024 } S02_Buffer { R_SIZE 1024 } M00_Buffer { R_SIZE 1024 } S03_Buffer { R_SIZE 1024 } S04_Buffer { R_SIZE 1024 } } }   } \
   CONFIG.NUM_SI {4} \
 ] $smartconnect_vmix_0

  # Create instance: smartconnect_vmix_1, and set properties
  set smartconnect_vmix_1 [ create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect:1.0 smartconnect_vmix_1 ]
  set_property -dict [ list \
   CONFIG.ADVANCED_PROPERTIES {    __view__ { functional { S07_Buffer { R_SIZE 1024 } S00_Buffer { R_SIZE 1024 } S06_Buffer { R_SIZE 1024 } S05_Buffer { R_SIZE 1024 } S01_Buffer { R_SIZE 1024 } S02_Buffer { R_SIZE 1024 } M00_Buffer { R_SIZE 1024 } S03_Buffer { R_SIZE 1024 } S04_Buffer { R_SIZE 1024 } } }   } \
   CONFIG.NUM_SI {4} \
 ] $smartconnect_vmix_1

  # Create instance: v_hdmi_tx_ss_0, and set properties
  set v_hdmi_tx_ss_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:v_hdmi_tx_ss:3.1 v_hdmi_tx_ss_0 ]
  set_property -dict [ list \
   CONFIG.C_INCLUDE_LOW_RESO_VID {true} \
   CONFIG.C_INCLUDE_YUV420_SUP {true} \
   CONFIG.C_INPUT_PIXELS_PER_CLOCK {4} \
   CONFIG.C_MAX_BITS_PER_COMPONENT {8} \
 ] $v_hdmi_tx_ss_0

  # Create instance: v_mix_0, and set properties
  set v_mix_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:v_mix:5.1 v_mix_0 ]
  set_property -dict [ list \
   CONFIG.AXIMM_ADDR_WIDTH {64} \
   CONFIG.AXIMM_BURST_LENGTH {256} \
   CONFIG.AXIMM_DATA_WIDTH {256} \
   CONFIG.C_M_AXI_MM_VIDEO10_DATA_WIDTH {256} \
   CONFIG.C_M_AXI_MM_VIDEO11_DATA_WIDTH {256} \
   CONFIG.C_M_AXI_MM_VIDEO12_DATA_WIDTH {256} \
   CONFIG.C_M_AXI_MM_VIDEO13_DATA_WIDTH {256} \
   CONFIG.C_M_AXI_MM_VIDEO14_DATA_WIDTH {256} \
   CONFIG.C_M_AXI_MM_VIDEO15_DATA_WIDTH {256} \
   CONFIG.C_M_AXI_MM_VIDEO16_DATA_WIDTH {256} \
   CONFIG.C_M_AXI_MM_VIDEO1_DATA_WIDTH {256} \
   CONFIG.C_M_AXI_MM_VIDEO2_DATA_WIDTH {256} \
   CONFIG.C_M_AXI_MM_VIDEO3_DATA_WIDTH {256} \
   CONFIG.C_M_AXI_MM_VIDEO4_DATA_WIDTH {256} \
   CONFIG.C_M_AXI_MM_VIDEO5_DATA_WIDTH {256} \
   CONFIG.C_M_AXI_MM_VIDEO6_DATA_WIDTH {256} \
   CONFIG.C_M_AXI_MM_VIDEO7_DATA_WIDTH {256} \
   CONFIG.C_M_AXI_MM_VIDEO8_DATA_WIDTH {256} \
   CONFIG.C_M_AXI_MM_VIDEO9_DATA_WIDTH {256} \
   CONFIG.LAYER10_ALPHA {true} \
   CONFIG.LAYER10_VIDEO_FORMAT {10} \
   CONFIG.LAYER11_ALPHA {true} \
   CONFIG.LAYER11_VIDEO_FORMAT {10} \
   CONFIG.LAYER12_ALPHA {true} \
   CONFIG.LAYER12_VIDEO_FORMAT {10} \
   CONFIG.LAYER13_ALPHA {true} \
   CONFIG.LAYER13_VIDEO_FORMAT {26} \
   CONFIG.LAYER1_ALPHA {true} \
   CONFIG.LAYER1_VIDEO_FORMAT {20} \
   CONFIG.LAYER2_ALPHA {true} \
   CONFIG.LAYER2_VIDEO_FORMAT {20} \
   CONFIG.LAYER3_ALPHA {true} \
   CONFIG.LAYER3_VIDEO_FORMAT {20} \
   CONFIG.LAYER4_ALPHA {true} \
   CONFIG.LAYER4_VIDEO_FORMAT {20} \
   CONFIG.LAYER5_ALPHA {true} \
   CONFIG.LAYER5_VIDEO_FORMAT {12} \
   CONFIG.LAYER6_ALPHA {true} \
   CONFIG.LAYER6_VIDEO_FORMAT {12} \
   CONFIG.LAYER7_ALPHA {true} \
   CONFIG.LAYER7_VIDEO_FORMAT {12} \
   CONFIG.LAYER8_ALPHA {true} \
   CONFIG.LAYER8_VIDEO_FORMAT {12} \
   CONFIG.LAYER9_ALPHA {true} \
   CONFIG.LAYER9_VIDEO_FORMAT {26} \
   CONFIG.MAX_DATA_WIDTH {8} \
   CONFIG.NR_LAYERS {10} \
   CONFIG.SAMPLES_PER_CLOCK {4} \
 ] $v_mix_0

  # Create instance: xlconstant_0, and set properties
  set xlconstant_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:xlconstant:1.1 xlconstant_0 ]
  set_property -dict [ list \
   CONFIG.CONST_VAL {0} \
   CONFIG.CONST_WIDTH {96} \
 ] $xlconstant_0

  # Create instance: xlslice_20, and set properties
  set xlslice_20 [ create_bd_cell -type ip -vlnv xilinx.com:ip:xlslice:1.0 xlslice_20 ]
  set_property -dict [ list \
   CONFIG.DIN_FROM {0} \
   CONFIG.DIN_TO {0} \
   CONFIG.DOUT_WIDTH {1} \
 ] $xlslice_20

  # Create interface connections
  connect_bd_intf_net -intf_net Conn1 [get_bd_intf_pins AUDIO_IN] [get_bd_intf_pins v_hdmi_tx_ss_0/AUDIO_IN]
  connect_bd_intf_net -intf_net Conn3 [get_bd_intf_pins HDMI_CTL_IIC] [get_bd_intf_pins hdmi_tx_phy/HDMI_CTL_IIC]
  connect_bd_intf_net -intf_net Conn6 [get_bd_intf_pins S_AXI] [get_bd_intf_pins hdmi_tx_phy/S_AXI]
  connect_bd_intf_net -intf_net Conn8 [get_bd_intf_pins axi4lite] [get_bd_intf_pins hdmi_tx_phy/axi4lite]
  connect_bd_intf_net -intf_net S_AXI_CPU_IN_1 [get_bd_intf_pins S_AXI_CPU_IN] [get_bd_intf_pins v_hdmi_tx_ss_0/S_AXI_CPU_IN]
  connect_bd_intf_net -intf_net hdmi_tx_pipe_M_AXIS [get_bd_intf_pins hdmi_tx_phy/M_AXIS] [get_bd_intf_pins v_hdmi_tx_ss_0/VIDEO_IN]
  connect_bd_intf_net -intf_net hdmi_tx_pipe_status_sb_tx [get_bd_intf_pins hdmi_tx_phy/status_sb_tx] [get_bd_intf_pins v_hdmi_tx_ss_0/SB_STATUS_IN]
  connect_bd_intf_net -intf_net s_axi_lite_vmix_1 [get_bd_intf_pins s_axi_ctrl_vmix] [get_bd_intf_pins v_mix_0/s_axi_CTRL]
  connect_bd_intf_net -intf_net smartconnect_0_M00_AXI [get_bd_intf_pins vmix_mm_axi_vid_rd_0] [get_bd_intf_pins smartconnect_vmix_0/M00_AXI]
  connect_bd_intf_net -intf_net smartconnect_1_M00_AXI [get_bd_intf_pins vmix_mm_axi_vid_rd_1] [get_bd_intf_pins smartconnect_vmix_1/M00_AXI]
  connect_bd_intf_net -intf_net v_hdmi_tx_ss_0_DDC_OUT [get_bd_intf_pins TX_DDC_OUT] [get_bd_intf_pins v_hdmi_tx_ss_0/DDC_OUT]
  connect_bd_intf_net -intf_net v_hdmi_tx_ss_0_LINK_DATA0_OUT [get_bd_intf_pins hdmi_tx_phy/tx_axi4s_ch0] [get_bd_intf_pins v_hdmi_tx_ss_0/LINK_DATA0_OUT]
  connect_bd_intf_net -intf_net v_hdmi_tx_ss_0_LINK_DATA1_OUT [get_bd_intf_pins hdmi_tx_phy/tx_axi4s_ch1] [get_bd_intf_pins v_hdmi_tx_ss_0/LINK_DATA1_OUT]
  connect_bd_intf_net -intf_net v_hdmi_tx_ss_0_LINK_DATA2_OUT [get_bd_intf_pins hdmi_tx_phy/tx_axi4s_ch2] [get_bd_intf_pins v_hdmi_tx_ss_0/LINK_DATA2_OUT]
  connect_bd_intf_net -intf_net v_mix_0_m_axi_mm_video1 [get_bd_intf_pins smartconnect_vmix_0/S00_AXI] [get_bd_intf_pins v_mix_0/m_axi_mm_video1]
  connect_bd_intf_net -intf_net v_mix_0_m_axi_mm_video2 [get_bd_intf_pins smartconnect_vmix_0/S01_AXI] [get_bd_intf_pins v_mix_0/m_axi_mm_video2]
  connect_bd_intf_net -intf_net v_mix_0_m_axi_mm_video3 [get_bd_intf_pins smartconnect_vmix_1/S00_AXI] [get_bd_intf_pins v_mix_0/m_axi_mm_video3]
  connect_bd_intf_net -intf_net v_mix_0_m_axi_mm_video4 [get_bd_intf_pins smartconnect_vmix_1/S01_AXI] [get_bd_intf_pins v_mix_0/m_axi_mm_video4]
  connect_bd_intf_net -intf_net v_mix_0_m_axi_mm_video5 [get_bd_intf_pins smartconnect_vmix_0/S02_AXI] [get_bd_intf_pins v_mix_0/m_axi_mm_video5]
  connect_bd_intf_net -intf_net v_mix_0_m_axi_mm_video6 [get_bd_intf_pins smartconnect_vmix_0/S03_AXI] [get_bd_intf_pins v_mix_0/m_axi_mm_video6]
  connect_bd_intf_net -intf_net v_mix_0_m_axi_mm_video7 [get_bd_intf_pins smartconnect_vmix_1/S02_AXI] [get_bd_intf_pins v_mix_0/m_axi_mm_video7]
  connect_bd_intf_net -intf_net v_mix_0_m_axi_mm_video8 [get_bd_intf_pins smartconnect_vmix_1/S03_AXI] [get_bd_intf_pins v_mix_0/m_axi_mm_video8]
  connect_bd_intf_net -intf_net v_mix_0_m_axi_mm_video9 [get_bd_intf_pins vmix_mm_axi_vid_rd_2] [get_bd_intf_pins v_mix_0/m_axi_mm_video9]
  connect_bd_intf_net -intf_net v_mix_0_m_axis_video [get_bd_intf_pins hdmi_tx_phy/s_axis_video] [get_bd_intf_pins v_mix_0/m_axis_video]

  # Create port connections
  connect_bd_net -net ARESETN_2 [get_bd_pins ARESETN] [get_bd_pins hdmi_tx_phy/ARESETN] [get_bd_pins v_hdmi_tx_ss_0/s_axi_cpu_aresetn]
  connect_bd_net -net IDT_8T49N241_LOL_IN_1 [get_bd_pins IDT_8T49N241_LOL_IN] [get_bd_pins hdmi_tx_phy/IDT_8T49N241_LOL_IN]
  connect_bd_net -net RX_DATA_IN_rxn_1 [get_bd_pins RX_DATA_IN_rxn] [get_bd_pins hdmi_tx_phy/RX_DATA_IN_rxn]
  connect_bd_net -net RX_DATA_IN_rxp_1 [get_bd_pins RX_DATA_IN_rxp] [get_bd_pins hdmi_tx_phy/RX_DATA_IN_rxp]
  connect_bd_net -net TX_HPD_IN_1 [get_bd_pins TX_HPD_IN] [get_bd_pins v_hdmi_tx_ss_0/hpd]
  connect_bd_net -net TX_REFCLK_N_IN_1 [get_bd_pins TX_REFCLK_N_IN] [get_bd_pins hdmi_tx_phy/TX_REFCLK_N_IN]
  connect_bd_net -net TX_REFCLK_P_IN_1 [get_bd_pins TX_REFCLK_P_IN] [get_bd_pins hdmi_tx_phy/TX_REFCLK_P_IN]
  connect_bd_net -net acr_cts_1 [get_bd_pins acr_cts] [get_bd_pins v_hdmi_tx_ss_0/acr_cts]
  connect_bd_net -net acr_n_1 [get_bd_pins acr_n] [get_bd_pins v_hdmi_tx_ss_0/acr_n]
  connect_bd_net -net acr_valid_1 [get_bd_pins acr_valid] [get_bd_pins v_hdmi_tx_ss_0/acr_valid]
  connect_bd_net -net altclk_1 [get_bd_pins altclk] [get_bd_pins hdmi_tx_phy/altclk] [get_bd_pins v_hdmi_tx_ss_0/s_axi_cpu_aclk]
  connect_bd_net -net aresetn1_1 [get_bd_pins aresetn1] [get_bd_pins hdmi_tx_phy/aresetn1] [get_bd_pins v_hdmi_tx_ss_0/s_axis_video_aresetn]
  connect_bd_net -net aresetn_1 [get_bd_pins sc_aresetn] [get_bd_pins smartconnect_vmix_0/aresetn] [get_bd_pins smartconnect_vmix_1/aresetn]
  connect_bd_net -net clk_wizard_0_clk_out3 [get_bd_pins s_axis_aclk] [get_bd_pins hdmi_tx_phy/aclk] [get_bd_pins smartconnect_vmix_0/aclk] [get_bd_pins smartconnect_vmix_1/aclk] [get_bd_pins v_hdmi_tx_ss_0/s_axis_video_aclk] [get_bd_pins v_mix_0/ap_clk]
  connect_bd_net -net hdmi_tx_phy_tx_tmds_clk [get_bd_pins tx_tmds_clk] [get_bd_pins hdmi_tx_phy/tx_tmds_clk]
  connect_bd_net -net hdmi_tx_pipe_TX_EN_OUT [get_bd_pins TX_EN_OUT] [get_bd_pins hdmi_tx_phy/TX_EN_OUT]
  connect_bd_net -net hdmi_tx_pipe_iic2intc_irpt [get_bd_pins iic2intc_irpt] [get_bd_pins hdmi_tx_phy/iic2intc_irpt]
  connect_bd_net -net hdmi_tx_pipe_irq [get_bd_pins irq] [get_bd_pins hdmi_tx_phy/irq]
  connect_bd_net -net hdmi_tx_pipe_tx_usrclk [get_bd_pins hdmi_tx_phy/tx_usrclk] [get_bd_pins v_hdmi_tx_ss_0/link_clk]
  connect_bd_net -net hdmi_tx_pipe_tx_video_clk [get_bd_pins hdmi_tx_phy/tx_video_clk] [get_bd_pins v_hdmi_tx_ss_0/video_clk]
  connect_bd_net -net hdmi_tx_pipe_txn_0 [get_bd_pins TX_DATA_OUT_txn] [get_bd_pins hdmi_tx_phy/TX_DATA_OUT_txn]
  connect_bd_net -net hdmi_tx_pipe_txp_0 [get_bd_pins TX_DATA_OUT_txp] [get_bd_pins hdmi_tx_phy/TX_DATA_OUT_txp]
  connect_bd_net -net ps_gpio_1 [get_bd_pins Din] [get_bd_pins xlslice_20/Din]
  connect_bd_net -net s_axis_audio_aclk_1 [get_bd_pins s_axis_audio_aclk] [get_bd_pins v_hdmi_tx_ss_0/s_axis_audio_aclk]
  connect_bd_net -net s_axis_audio_aresetn_1 [get_bd_pins s_axis_audio_aresetn] [get_bd_pins v_hdmi_tx_ss_0/s_axis_audio_aresetn]
  connect_bd_net -net v_hdmi_tx_ss_0_irq [get_bd_pins irq1] [get_bd_pins v_hdmi_tx_ss_0/irq]
  connect_bd_net -net v_mix_0_interrupt [get_bd_pins vmix_intr] [get_bd_pins v_mix_0/interrupt]
  connect_bd_net -net xlconstant_0_dout [get_bd_pins v_mix_0/s_axis_video_TDATA] [get_bd_pins v_mix_0/s_axis_video_TVALID] [get_bd_pins xlconstant_0/dout]
  connect_bd_net -net xlslice_20_Dout [get_bd_pins v_mix_0/ap_rst_n] [get_bd_pins xlslice_20/Dout]

  # Restore current instance
  current_bd_instance $oldCurInst
}

# Hierarchical cell: audio_pipe
proc create_hier_cell_audio_pipe { parentCell nameHier } {

  variable script_folder

  if { $parentCell eq "" || $nameHier eq "" } {
     catch {common::send_gid_msg -ssname BD::TCL -id 2092 -severity "ERROR" "create_hier_cell_audio_pipe() - Empty argument(s)!"}
     return
  }

  # Get object for parentCell
  set parentObj [get_bd_cells $parentCell]
  if { $parentObj == "" } {
     catch {common::send_gid_msg -ssname BD::TCL -id 2090 -severity "ERROR" "Unable to find parent cell <$parentCell>!"}
     return
  }

  # Make sure parentObj is hier blk
  set parentType [get_property TYPE $parentObj]
  if { $parentType ne "hier" } {
     catch {common::send_gid_msg -ssname BD::TCL -id 2091 -severity "ERROR" "Parent <$parentObj> has TYPE = <$parentType>. Expected to be <hier>."}
     return
  }

  # Save current instance; Restore later
  set oldCurInst [current_bd_instance .]

  # Set parent object as current
  current_bd_instance $parentObj

  # Create cell and set as current instance
  set hier_obj [create_bd_cell -type hier $nameHier]
  current_bd_instance $hier_obj

  # Create interface pins
  create_bd_intf_pin -mode Slave -vlnv xilinx.com:interface:diff_clock_rtl:1.0 CLK_IN_AUDIO

  create_bd_intf_pin -mode Master -vlnv xilinx.com:interface:aximm_rtl:1.0 m_axi_mm2s

  create_bd_intf_pin -mode Master -vlnv xilinx.com:interface:axis_rtl:1.0 m_axis_mm2s

  create_bd_intf_pin -mode Slave -vlnv xilinx.com:interface:aximm_rtl:1.0 s_axi_ctrl_acr

  create_bd_intf_pin -mode Slave -vlnv xilinx.com:interface:aximm_rtl:1.0 s_axi_ctrl_aud_for


  # Create pins
  create_bd_pin -dir O -from 19 -to 0 aud_acr_cts_out
  create_bd_pin -dir O -from 19 -to 0 aud_acr_n_out
  create_bd_pin -dir O aud_acr_valid_out
  create_bd_pin -dir O -from 0 -to 0 -type clk aud_clk
  create_bd_pin -dir O -type rst aud_resetn_out
  create_bd_pin -dir I -type rst ext_reset_in
  create_bd_pin -dir I -type clk hdmi_clk
  create_bd_pin -dir O -type intr irq_mm2s
  create_bd_pin -dir I -type clk s_axi_lite_aclk
  create_bd_pin -dir I -type rst s_axi_lite_aresetn

  # Create instance: audio_formatter_0, and set properties
  set audio_formatter_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:audio_formatter:1.0 audio_formatter_0 ]
  set_property -dict [ list \
   CONFIG.C_INCLUDE_S2MM {0} \
 ] $audio_formatter_0

  # Create instance: hdmi_acr_ctrl_0, and set properties
  set hdmi_acr_ctrl_0 [ create_bd_cell -type ip -vlnv xilinx.com:user:hdmi_acr_ctrl:1.1 hdmi_acr_ctrl_0 ]

  # Create instance: proc_sys_reset_aud_clk, and set properties
  set proc_sys_reset_aud_clk [ create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 proc_sys_reset_aud_clk ]

  # Create instance: refclk_aud
  create_hier_cell_refclk_aud $hier_obj refclk_aud

  # Create instance: xlconstant_0, and set properties
  set xlconstant_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:xlconstant:1.1 xlconstant_0 ]
  set_property -dict [ list \
   CONFIG.CONST_VAL {0} \
   CONFIG.CONST_WIDTH {20} \
 ] $xlconstant_0

  # Create instance: xlconstant_1, and set properties
  set xlconstant_1 [ create_bd_cell -type ip -vlnv xilinx.com:ip:xlconstant:1.1 xlconstant_1 ]

  # Create interface connections
  connect_bd_intf_net -intf_net Conn1 [get_bd_intf_pins m_axi_mm2s] [get_bd_intf_pins audio_formatter_0/m_axi_mm2s]
  connect_bd_intf_net -intf_net Conn2 [get_bd_intf_pins s_axi_ctrl_acr] [get_bd_intf_pins hdmi_acr_ctrl_0/axi]
  connect_bd_intf_net -intf_net Conn3 [get_bd_intf_pins CLK_IN_AUDIO] [get_bd_intf_pins refclk_aud/CLK_IN_D_0]
  connect_bd_intf_net -intf_net audio_formatter_0_m_axis_mm2s [get_bd_intf_pins m_axis_mm2s] [get_bd_intf_pins audio_formatter_0/m_axis_mm2s]
  connect_bd_intf_net -intf_net smartconnect_gp0_M08_AXI [get_bd_intf_pins s_axi_ctrl_aud_for] [get_bd_intf_pins audio_formatter_0/s_axi_lite]

  # Create port connections
  connect_bd_net -net audio_formatter_0_irq_mm2s [get_bd_pins irq_mm2s] [get_bd_pins audio_formatter_0/irq_mm2s]
  connect_bd_net -net clk_wiz_clk_out2 [get_bd_pins s_axi_lite_aclk] [get_bd_pins audio_formatter_0/s_axi_lite_aclk] [get_bd_pins hdmi_acr_ctrl_0/axi_aclk]
  connect_bd_net -net ext_reset_in1_1 [get_bd_pins ext_reset_in] [get_bd_pins proc_sys_reset_aud_clk/ext_reset_in]
  connect_bd_net -net hdmi_acr_ctrl_0_aud_acr_cts_out [get_bd_pins aud_acr_cts_out] [get_bd_pins hdmi_acr_ctrl_0/aud_acr_cts_out]
  connect_bd_net -net hdmi_acr_ctrl_0_aud_acr_n_out [get_bd_pins aud_acr_n_out] [get_bd_pins hdmi_acr_ctrl_0/aud_acr_n_out]
  connect_bd_net -net hdmi_acr_ctrl_0_aud_acr_valid_out [get_bd_pins aud_acr_valid_out] [get_bd_pins hdmi_acr_ctrl_0/aud_acr_valid_out]
  connect_bd_net -net hdmi_acr_ctrl_0_aud_resetn_out [get_bd_pins aud_resetn_out] [get_bd_pins hdmi_acr_ctrl_0/aud_resetn_out]
  connect_bd_net -net hdmi_clk_1 [get_bd_pins hdmi_clk] [get_bd_pins hdmi_acr_ctrl_0/hdmi_clk]
  connect_bd_net -net proc_sys_reset_aud_clk_peripheral_aresetn [get_bd_pins audio_formatter_0/m_axis_mm2s_aresetn] [get_bd_pins proc_sys_reset_aud_clk/peripheral_aresetn]
  connect_bd_net -net proc_sys_reset_aud_clk_peripheral_reset [get_bd_pins audio_formatter_0/aud_mreset] [get_bd_pins proc_sys_reset_aud_clk/peripheral_reset]
  connect_bd_net -net refclk_aud_BUFG_O [get_bd_pins aud_clk] [get_bd_pins audio_formatter_0/aud_mclk] [get_bd_pins audio_formatter_0/m_axis_mm2s_aclk] [get_bd_pins hdmi_acr_ctrl_0/aud_clk] [get_bd_pins proc_sys_reset_aud_clk/slowest_sync_clk] [get_bd_pins refclk_aud/BUFG_O]
  connect_bd_net -net rst_processor_1_100M_peripheral_aresetn [get_bd_pins s_axi_lite_aresetn] [get_bd_pins audio_formatter_0/s_axi_lite_aresetn] [get_bd_pins hdmi_acr_ctrl_0/axi_aresetn]
  connect_bd_net -net xlconstant_0_dout [get_bd_pins hdmi_acr_ctrl_0/pll_lock_in] [get_bd_pins xlconstant_1/dout]
  connect_bd_net -net xlconstant_0_dout1 [get_bd_pins hdmi_acr_ctrl_0/aud_acr_cts_in] [get_bd_pins hdmi_acr_ctrl_0/aud_acr_n_in] [get_bd_pins hdmi_acr_ctrl_0/aud_acr_valid_in] [get_bd_pins xlconstant_0/dout]

  # Restore current instance
  current_bd_instance $oldCurInst
}


# Procedure to create entire design; Provide argument to make
# procedure reusable. If parentCell is "", will use root.
proc create_root_design { parentCell } {

  variable script_folder
  variable design_name

  if { $parentCell eq "" } {
     set parentCell [get_bd_cells /]
  }

  # Get object for parentCell
  set parentObj [get_bd_cells $parentCell]
  if { $parentObj == "" } {
     catch {common::send_gid_msg -ssname BD::TCL -id 2090 -severity "ERROR" "Unable to find parent cell <$parentCell>!"}
     return
  }

  # Make sure parentObj is hier blk
  set parentType [get_property TYPE $parentObj]
  if { $parentType ne "hier" } {
     catch {common::send_gid_msg -ssname BD::TCL -id 2091 -severity "ERROR" "Parent <$parentObj> has TYPE = <$parentType>. Expected to be <hier>."}
     return
  }

  # Save current instance; Restore later
  set oldCurInst [current_bd_instance .]

  # Set parent object as current
  current_bd_instance $parentObj


  # Create interface ports
  set CH0_LPDDR4_0_0 [ create_bd_intf_port -mode Master -vlnv xilinx.com:interface:lpddr4_rtl:1.0 CH0_LPDDR4_0_0 ]

  set CH0_LPDDR4_1_0 [ create_bd_intf_port -mode Master -vlnv xilinx.com:interface:lpddr4_rtl:1.0 CH0_LPDDR4_1_0 ]

  set CH1_LPDDR4_0_0 [ create_bd_intf_port -mode Master -vlnv xilinx.com:interface:lpddr4_rtl:1.0 CH1_LPDDR4_0_0 ]

  set CH1_LPDDR4_1_0 [ create_bd_intf_port -mode Master -vlnv xilinx.com:interface:lpddr4_rtl:1.0 CH1_LPDDR4_1_0 ]

  set CLK_IN_AUDIO [ create_bd_intf_port -mode Slave -vlnv xilinx.com:interface:diff_clock_rtl:1.0 CLK_IN_AUDIO ]

  set HDMI_CTL_IIC [ create_bd_intf_port -mode Master -vlnv xilinx.com:interface:iic_rtl:1.0 HDMI_CTL_IIC ]

  set TX_DDC_OUT [ create_bd_intf_port -mode Master -vlnv xilinx.com:interface:iic_rtl:1.0 TX_DDC_OUT ]

  set csi_mipi_phy_if [ create_bd_intf_port -mode Slave -vlnv xilinx.com:interface:mipi_phy_rtl:1.0 csi_mipi_phy_if ]

  set sensor_iic [ create_bd_intf_port -mode Master -vlnv xilinx.com:interface:iic_rtl:1.0 sensor_iic ]

  set sys_clk0_0 [ create_bd_intf_port -mode Slave -vlnv xilinx.com:interface:diff_clock_rtl:1.0 sys_clk0_0 ]
  set_property -dict [ list \
   CONFIG.FREQ_HZ {201500000} \
   ] $sys_clk0_0

  set sys_clk1_0 [ create_bd_intf_port -mode Slave -vlnv xilinx.com:interface:diff_clock_rtl:1.0 sys_clk1_0 ]
  set_property -dict [ list \
   CONFIG.FREQ_HZ {201500000} \
   ] $sys_clk1_0


  # Create ports
  set IDT_8T49N241_LOL_IN [ create_bd_port -dir I IDT_8T49N241_LOL_IN ]
  set RX_DATA_IN_rxn [ create_bd_port -dir I -from 3 -to 0 RX_DATA_IN_rxn ]
  set RX_DATA_IN_rxp [ create_bd_port -dir I -from 3 -to 0 RX_DATA_IN_rxp ]
  set TX_DATA_OUT_txn [ create_bd_port -dir O -from 3 -to 0 TX_DATA_OUT_txn ]
  set TX_DATA_OUT_txp [ create_bd_port -dir O -from 3 -to 0 TX_DATA_OUT_txp ]
  set TX_EN_OUT [ create_bd_port -dir O -from 0 -to 0 TX_EN_OUT ]
  set TX_HPD_IN [ create_bd_port -dir I TX_HPD_IN ]
  set TX_REFCLK_N_IN [ create_bd_port -dir I TX_REFCLK_N_IN ]
  set TX_REFCLK_P_IN [ create_bd_port -dir I TX_REFCLK_P_IN ]
  set sensor_gpio_flash [ create_bd_port -dir O -from 0 -to 0 sensor_gpio_flash ]
  set sensor_gpio_rst [ create_bd_port -dir O -from 0 -to 0 sensor_gpio_rst ]
  set sensor_gpio_spi_cs_n [ create_bd_port -dir O -from 0 -to 0 sensor_gpio_spi_cs_n ]

  # Create instance: CIPS_0, and set properties
  set CIPS_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:versal_cips:2.1 CIPS_0 ]
  set_property -dict [ list \
   CONFIG.CPM_PCIE0_MODES {None} \
   CONFIG.CPM_PCIE1_MODES {None} \
   CONFIG.PMC_CRP_CFU_REF_CTRL_DIVISOR0 {4} \
   CONFIG.PMC_CRP_CFU_REF_CTRL_SRCSEL {PPLL} \
   CONFIG.PMC_CRP_DFT_OSC_REF_CTRL_DIVISOR0 {3} \
   CONFIG.PMC_CRP_DFT_OSC_REF_CTRL_SRCSEL {PPLL} \
   CONFIG.PMC_CRP_HSM0_REF_CTRL_DIVISOR0 {36} \
   CONFIG.PMC_CRP_HSM0_REF_CTRL_SRCSEL {PPLL} \
   CONFIG.PMC_CRP_HSM1_REF_CTRL_DIVISOR0 {9} \
   CONFIG.PMC_CRP_HSM1_REF_CTRL_SRCSEL {PPLL} \
   CONFIG.PMC_CRP_I2C_REF_CTRL_DIVISOR0 {12} \
   CONFIG.PMC_CRP_I2C_REF_CTRL_SRCSEL {PPLL} \
   CONFIG.PMC_CRP_LSBUS_REF_CTRL_DIVISOR0 {12} \
   CONFIG.PMC_CRP_LSBUS_REF_CTRL_SRCSEL {PPLL} \
   CONFIG.PMC_CRP_NOC_REF_CTRL_SRCSEL {NPLL} \
   CONFIG.PMC_CRP_NPI_REF_CTRL_DIVISOR0 {4} \
   CONFIG.PMC_CRP_NPI_REF_CTRL_SRCSEL {PPLL} \
   CONFIG.PMC_CRP_NPLL_CTRL_CLKOUTDIV {4} \
   CONFIG.PMC_CRP_NPLL_CTRL_FBDIV {115} \
   CONFIG.PMC_CRP_NPLL_CTRL_SRCSEL {REF_CLK} \
   CONFIG.PMC_CRP_NPLL_TO_XPD_CTRL_DIVISOR0 {1} \
   CONFIG.PMC_CRP_OSPI_REF_CTRL_DIVISOR0 {4} \
   CONFIG.PMC_CRP_OSPI_REF_CTRL_SRCSEL {PPLL} \
   CONFIG.PMC_CRP_PL0_REF_CTRL_DIVISOR0 {12} \
   CONFIG.PMC_CRP_PL0_REF_CTRL_SRCSEL {PPLL} \
   CONFIG.PMC_CRP_PL1_REF_CTRL_DIVISOR0 {3} \
   CONFIG.PMC_CRP_PL1_REF_CTRL_SRCSEL {NPLL} \
   CONFIG.PMC_CRP_PL2_REF_CTRL_DIVISOR0 {3} \
   CONFIG.PMC_CRP_PL2_REF_CTRL_SRCSEL {NPLL} \
   CONFIG.PMC_CRP_PL3_REF_CTRL_DIVISOR0 {3} \
   CONFIG.PMC_CRP_PL3_REF_CTRL_SRCSEL {NPLL} \
   CONFIG.PMC_CRP_PPLL_CTRL_CLKOUTDIV {2} \
   CONFIG.PMC_CRP_PPLL_CTRL_FBDIV {72} \
   CONFIG.PMC_CRP_PPLL_CTRL_SRCSEL {REF_CLK} \
   CONFIG.PMC_CRP_PPLL_TO_XPD_CTRL_DIVISOR0 {2} \
   CONFIG.PMC_CRP_QSPI_REF_CTRL_DIVISOR0 {5} \
   CONFIG.PMC_CRP_QSPI_REF_CTRL_SRCSEL {PPLL} \
   CONFIG.PMC_CRP_SDIO0_REF_CTRL_DIVISOR0 {6} \
   CONFIG.PMC_CRP_SDIO0_REF_CTRL_SRCSEL {PPLL} \
   CONFIG.PMC_CRP_SDIO1_REF_CTRL_DIVISOR0 {6} \
   CONFIG.PMC_CRP_SDIO1_REF_CTRL_SRCSEL {PPLL} \
   CONFIG.PMC_CRP_SD_DLL_REF_CTRL_DIVISOR0 {1} \
   CONFIG.PMC_CRP_SD_DLL_REF_CTRL_SRCSEL {PPLL} \
   CONFIG.PMC_CRP_SYSMON_REF_CTRL_SRCSEL {NPI_REF_CLK} \
   CONFIG.PMC_CRP_TEST_PATTERN_REF_CTRL_DIVISOR0 {6} \
   CONFIG.PMC_CRP_TEST_PATTERN_REF_CTRL_SRCSEL {PPLL} \
   CONFIG.PMC_GPIO0_MIO_PERIPHERAL_ENABLE {1} \
   CONFIG.PMC_GPIO1_MIO_PERIPHERAL_ENABLE {1} \
   CONFIG.PMC_HSM0_CLOCK_ENABLE {1} \
   CONFIG.PMC_HSM1_CLOCK_ENABLE {1} \
   CONFIG.PMC_MIO_37_DIRECTION {out} \
   CONFIG.PMC_MIO_37_OUTPUT_DATA {high} \
   CONFIG.PMC_MIO_37_PULL {pulldown} \
   CONFIG.PMC_MIO_37_USAGE {GPIO} \
   CONFIG.PMC_MIO_48_DIRECTION {out} \
   CONFIG.PMC_MIO_48_PULL {pullup} \
   CONFIG.PMC_MIO_48_USAGE {GPIO} \
   CONFIG.PMC_MIO_49_DIRECTION {out} \
   CONFIG.PMC_MIO_49_PULL {pullup} \
   CONFIG.PMC_MIO_49_USAGE {GPIO} \
   CONFIG.PMC_QSPI_GRP_FBCLK_ENABLE {1} \
   CONFIG.PMC_QSPI_PERIPHERAL_DATA_MODE {x4} \
   CONFIG.PMC_QSPI_PERIPHERAL_ENABLE {1} \
   CONFIG.PMC_QSPI_PERIPHERAL_MODE {Dual Parallel} \
   CONFIG.PMC_SD1_DATA_TRANSFER_MODE {8Bit} \
   CONFIG.PMC_SD1_GRP_CD_ENABLE {1} \
   CONFIG.PMC_SD1_GRP_POW_ENABLE {1} \
   CONFIG.PMC_SD1_PERIPHERAL_ENABLE {1} \
   CONFIG.PMC_SD1_PERIPHERAL_IO {PMC_MIO 26 .. 36} \
   CONFIG.PMC_SD1_SLOT_TYPE {SD 3.0} \
   CONFIG.PMC_USE_PMC_NOC_AXI0 {1} \
   CONFIG.PSPMC_MANUAL_CLOCK_ENABLE {1} \
   CONFIG.PS_CAN1_PERIPHERAL_ENABLE {1} \
   CONFIG.PS_CAN1_PERIPHERAL_IO {PMC_MIO 40 .. 41} \
   CONFIG.PS_CRF_ACPU_CTRL_DIVISOR0 {1} \
   CONFIG.PS_CRF_ACPU_CTRL_SRCSEL {APLL} \
   CONFIG.PS_CRF_APLL_CTRL_CLKOUTDIV {2} \
   CONFIG.PS_CRF_APLL_CTRL_FBDIV {81} \
   CONFIG.PS_CRF_APLL_CTRL_SRCSEL {REF_CLK} \
   CONFIG.PS_CRF_APLL_TO_XPD_CTRL_DIVISOR0 {2} \
   CONFIG.PS_CRF_DBG_FPD_CTRL_DIVISOR0 {2} \
   CONFIG.PS_CRF_DBG_FPD_CTRL_SRCSEL {PPLL} \
   CONFIG.PS_CRF_DBG_TRACE_CTRL_DIVISOR0 {3} \
   CONFIG.PS_CRF_DBG_TRACE_CTRL_SRCSEL {PPLL} \
   CONFIG.PS_CRF_FPD_LSBUS_CTRL_DIVISOR0 {6} \
   CONFIG.PS_CRF_FPD_LSBUS_CTRL_SRCSEL {PPLL} \
   CONFIG.PS_CRF_FPD_TOP_SWITCH_CTRL_DIVISOR0 {2} \
   CONFIG.PS_CRF_FPD_TOP_SWITCH_CTRL_SRCSEL {APLL} \
   CONFIG.PS_CRL_CAN0_REF_CTRL_DIVISOR0 {12} \
   CONFIG.PS_CRL_CAN0_REF_CTRL_SRCSEL {PPLL} \
   CONFIG.PS_CRL_CAN1_REF_CTRL_DIVISOR0 {6} \
   CONFIG.PS_CRL_CAN1_REF_CTRL_SRCSEL {NPLL} \
   CONFIG.PS_CRL_CPM_TOPSW_REF_CTRL_DIVISOR0 {2} \
   CONFIG.PS_CRL_CPM_TOPSW_REF_CTRL_SRCSEL {NPLL} \
   CONFIG.PS_CRL_CPU_R5_CTRL_DIVISOR0 {2} \
   CONFIG.PS_CRL_CPU_R5_CTRL_SRCSEL {RPLL} \
   CONFIG.PS_CRL_DBG_LPD_CTRL_DIVISOR0 {2} \
   CONFIG.PS_CRL_DBG_LPD_CTRL_SRCSEL {PPLL} \
   CONFIG.PS_CRL_DBG_TSTMP_CTRL_DIVISOR0 {2} \
   CONFIG.PS_CRL_DBG_TSTMP_CTRL_SRCSEL {PPLL} \
   CONFIG.PS_CRL_GEM0_REF_CTRL_DIVISOR0 {6} \
   CONFIG.PS_CRL_GEM0_REF_CTRL_SRCSEL {RPLL} \
   CONFIG.PS_CRL_GEM1_REF_CTRL_DIVISOR0 {6} \
   CONFIG.PS_CRL_GEM1_REF_CTRL_SRCSEL {RPLL} \
   CONFIG.PS_CRL_GEM_TSU_REF_CTRL_DIVISOR0 {3} \
   CONFIG.PS_CRL_GEM_TSU_REF_CTRL_SRCSEL {RPLL} \
   CONFIG.PS_CRL_I2C0_REF_CTRL_DIVISOR0 {6} \
   CONFIG.PS_CRL_I2C0_REF_CTRL_SRCSEL {PPLL} \
   CONFIG.PS_CRL_I2C1_REF_CTRL_DIVISOR0 {6} \
   CONFIG.PS_CRL_I2C1_REF_CTRL_SRCSEL {PPLL} \
   CONFIG.PS_CRL_IOU_SWITCH_CTRL_DIVISOR0 {3} \
   CONFIG.PS_CRL_IOU_SWITCH_CTRL_SRCSEL {RPLL} \
   CONFIG.PS_CRL_LPD_LSBUS_CTRL_DIVISOR0 {6} \
   CONFIG.PS_CRL_LPD_LSBUS_CTRL_SRCSEL {PPLL} \
   CONFIG.PS_CRL_LPD_TOP_SWITCH_CTRL_DIVISOR0 {2} \
   CONFIG.PS_CRL_LPD_TOP_SWITCH_CTRL_SRCSEL {RPLL} \
   CONFIG.PS_CRL_PSM_REF_CTRL_DIVISOR0 {2} \
   CONFIG.PS_CRL_PSM_REF_CTRL_SRCSEL {PPLL} \
   CONFIG.PS_CRL_RPLL_CTRL_CLKOUTDIV {4} \
   CONFIG.PS_CRL_RPLL_CTRL_FBDIV {90} \
   CONFIG.PS_CRL_RPLL_CTRL_SRCSEL {REF_CLK} \
   CONFIG.PS_CRL_RPLL_TO_XPD_CTRL_DIVISOR0 {3} \
   CONFIG.PS_CRL_SPI0_REF_CTRL_DIVISOR0 {6} \
   CONFIG.PS_CRL_SPI0_REF_CTRL_SRCSEL {PPLL} \
   CONFIG.PS_CRL_SPI1_REF_CTRL_DIVISOR0 {6} \
   CONFIG.PS_CRL_SPI1_REF_CTRL_SRCSEL {PPLL} \
   CONFIG.PS_CRL_TIMESTAMP_REF_CTRL_DIVISOR0 {6} \
   CONFIG.PS_CRL_TIMESTAMP_REF_CTRL_SRCSEL {PPLL} \
   CONFIG.PS_CRL_UART0_REF_CTRL_DIVISOR0 {6} \
   CONFIG.PS_CRL_UART0_REF_CTRL_SRCSEL {PPLL} \
   CONFIG.PS_CRL_UART1_REF_CTRL_DIVISOR0 {12} \
   CONFIG.PS_CRL_UART1_REF_CTRL_SRCSEL {PPLL} \
   CONFIG.PS_CRL_USB0_BUS_REF_CTRL_DIVISOR0 {30} \
   CONFIG.PS_CRL_USB0_BUS_REF_CTRL_SRCSEL {PPLL} \
   CONFIG.PS_CRL_USB3_DUAL_REF_CTRL_DIVISOR0 {100} \
   CONFIG.PS_CRL_USB3_DUAL_REF_CTRL_SRCSEL {PPLL} \
   CONFIG.PS_ENET0_GRP_MDIO_ENABLE {1} \
   CONFIG.PS_ENET0_GRP_MDIO_IO {PS_MIO 24 .. 25} \
   CONFIG.PS_ENET0_PERIPHERAL_ENABLE {1} \
   CONFIG.PS_ENET0_PERIPHERAL_IO {PS_MIO 0 .. 11} \
   CONFIG.PS_ENET1_PERIPHERAL_ENABLE {1} \
   CONFIG.PS_ENET1_PERIPHERAL_IO {PS_MIO 12 .. 23} \
   CONFIG.PS_GEM0_ROUTE_THROUGH_FPD {1} \
   CONFIG.PS_GEM1_ROUTE_THROUGH_FPD {1} \
   CONFIG.PS_GEN_IPI_0_ENABLE {1} \
   CONFIG.PS_GEN_IPI_0_MASTER {A72} \
   CONFIG.PS_GEN_IPI_1_ENABLE {1} \
   CONFIG.PS_GEN_IPI_1_MASTER {R5_0} \
   CONFIG.PS_GEN_IPI_2_ENABLE {1} \
   CONFIG.PS_GEN_IPI_2_MASTER {R5_1} \
   CONFIG.PS_GEN_IPI_3_ENABLE {1} \
   CONFIG.PS_GEN_IPI_3_MASTER {A72} \
   CONFIG.PS_GEN_IPI_4_ENABLE {1} \
   CONFIG.PS_GEN_IPI_4_MASTER {A72} \
   CONFIG.PS_GEN_IPI_5_ENABLE {1} \
   CONFIG.PS_GEN_IPI_5_MASTER {A72} \
   CONFIG.PS_GEN_IPI_6_ENABLE {1} \
   CONFIG.PS_GEN_IPI_6_MASTER {A72} \
   CONFIG.PS_GEN_IPI_PMCNOBUF_ENABLE {1} \
   CONFIG.PS_GEN_IPI_PMC_ENABLE {1} \
   CONFIG.PS_GEN_IPI_PSM_ENABLE {1} \
   CONFIG.PS_GPIO2_MIO_PERIPHERAL_ENABLE {1} \
   CONFIG.PS_GPIO_EMIO_PERIPHERAL_ENABLE {1} \
   CONFIG.PS_GPIO_EMIO_WIDTH {32} \
   CONFIG.PS_I2C0_PERIPHERAL_ENABLE {1} \
   CONFIG.PS_I2C0_PERIPHERAL_IO {PMC_MIO 46 .. 47} \
   CONFIG.PS_I2C1_PERIPHERAL_ENABLE {1} \
   CONFIG.PS_I2C1_PERIPHERAL_IO {PMC_MIO 44 .. 45} \
   CONFIG.PS_LPDMA0_ROUTE_THROUGH_FPD {1} \
   CONFIG.PS_LPDMA1_ROUTE_THROUGH_FPD {1} \
   CONFIG.PS_LPDMA2_ROUTE_THROUGH_FPD {1} \
   CONFIG.PS_LPDMA3_ROUTE_THROUGH_FPD {1} \
   CONFIG.PS_LPDMA4_ROUTE_THROUGH_FPD {1} \
   CONFIG.PS_LPDMA5_ROUTE_THROUGH_FPD {1} \
   CONFIG.PS_LPDMA6_ROUTE_THROUGH_FPD {1} \
   CONFIG.PS_LPDMA7_ROUTE_THROUGH_FPD {1} \
   CONFIG.PS_MIO_19_PULL {disable} \
   CONFIG.PS_MIO_21_PULL {disable} \
   CONFIG.PS_MIO_7_PULL {disable} \
   CONFIG.PS_MIO_9_PULL {disable} \
   CONFIG.PS_M_AXI_GP0_DATA_WIDTH {32} \
   CONFIG.PS_M_AXI_GP2_DATA_WIDTH {32} \
   CONFIG.PS_NUM_FABRIC_RESETS {4} \
   CONFIG.PS_TTC0_PERIPHERAL_ENABLE {1} \
   CONFIG.PS_TTC1_PERIPHERAL_ENABLE {1} \
   CONFIG.PS_TTC2_PERIPHERAL_ENABLE {1} \
   CONFIG.PS_TTC3_PERIPHERAL_ENABLE {1} \
   CONFIG.PS_UART0_BAUD_RATE {115200} \
   CONFIG.PS_UART0_PERIPHERAL_ENABLE {1} \
   CONFIG.PS_UART0_PERIPHERAL_IO {PMC_MIO 42 .. 43} \
   CONFIG.PS_USB3_PERIPHERAL_ENABLE {1} \
   CONFIG.PS_USB_ROUTE_THROUGH_FPD {1} \
   CONFIG.PS_USE_IRQ_0 {1} \
   CONFIG.PS_USE_IRQ_1 {1} \
   CONFIG.PS_USE_IRQ_2 {1} \
   CONFIG.PS_USE_IRQ_3 {1} \
   CONFIG.PS_USE_IRQ_4 {1} \
   CONFIG.PS_USE_IRQ_5 {1} \
   CONFIG.PS_USE_IRQ_6 {1} \
   CONFIG.PS_USE_IRQ_7 {1} \
   CONFIG.PS_USE_IRQ_8 {1} \
   CONFIG.PS_USE_IRQ_9 {1} \
   CONFIG.PS_USE_IRQ_10 {1} \
   CONFIG.PS_USE_IRQ_11 {1} \
   CONFIG.PS_USE_IRQ_12 {1} \
   CONFIG.PS_USE_IRQ_13 {1} \
   CONFIG.PS_USE_IRQ_14 {1} \
   CONFIG.PS_USE_IRQ_15 {1} \
   CONFIG.PS_USE_M_AXI_GP0 {1} \
   CONFIG.PS_USE_M_AXI_GP2 {1} \
   CONFIG.PS_USE_PMCPL_CLK0 {1} \
   CONFIG.PS_USE_PS_NOC_CCI {1} \
   CONFIG.PS_USE_PS_NOC_NCI_0 {1} \
   CONFIG.PS_USE_PS_NOC_NCI_1 {1} \
   CONFIG.PS_USE_PS_NOC_RPU_0 {1} \
   CONFIG.PS_WWDT0_CLOCK_IO {APB} \
   CONFIG.PS_WWDT0_PERIPHERAL_ENABLE {1} \
   CONFIG.PS_WWDT0_PERIPHERAL_IO {EMIO} \
 ] $CIPS_0

  # Create instance: NOC_0, and set properties
  set NOC_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_noc:1.0 NOC_0 ]
  set_property -dict [ list \
   CONFIG.CONTROLLERTYPE {LPDDR4_SDRAM} \
   CONFIG.MC0_FLIPPED_PINOUT {true} \
   CONFIG.MC1_FLIPPED_PINOUT {true} \
   CONFIG.MC_ADDR_BIT9 {CA6} \
   CONFIG.MC_BA_WIDTH {3} \
   CONFIG.MC_CHANNEL_INTERLEAVING {true} \
   CONFIG.MC_CHAN_REGION0 {DDR_LOW0} \
   CONFIG.MC_CHAN_REGION1 {DDR_CH1} \
   CONFIG.MC_CH_INTERLEAVING_SIZE {128_Bytes} \
   CONFIG.MC_COMPONENT_WIDTH {x32} \
   CONFIG.MC_DATAWIDTH {32} \
   CONFIG.MC_INPUTCLK0_PERIOD {4963} \
   CONFIG.MC_INTERLEAVE_SIZE {1024} \
   CONFIG.MC_LPDDR4_REFRESH_TYPE {PER_BANK} \
   CONFIG.MC_MEMORY_DEVICETYPE {Components} \
   CONFIG.MC_NO_CHANNELS {Dual} \
   CONFIG.MC_PRE_DEF_ADDR_MAP_SEL {ROW_BANK_COLUMN} \
   CONFIG.MC_RANK {1} \
   CONFIG.MC_REF_AND_PER_CAL_INTF {FALSE} \
   CONFIG.MC_ROWADDRESSWIDTH {16} \
   CONFIG.MC_TRC {60000} \
   CONFIG.NUM_CLKS {12} \
   CONFIG.NUM_MC {2} \
   CONFIG.NUM_MCP {4} \
   CONFIG.NUM_MI {1} \
   CONFIG.NUM_NMI {0} \
   CONFIG.NUM_NSI {0} \
   CONFIG.NUM_SI {15} \
 ] $NOC_0

  set_property -dict [ list \
   CONFIG.DATA_WIDTH {128} \
   CONFIG.CATEGORY {aie} \
 ] [get_bd_intf_pins /NOC_0/M00_AXI]

  set_property -dict [ list \
   CONFIG.DATA_WIDTH {128} \
   CONFIG.CONNECTIONS {MC_0 { read_bw {5} write_bw {5} read_avg_burst {4} write_avg_burst {4}} M00_AXI { read_bw {5} write_bw {5} read_avg_burst {4} write_avg_burst {4}} } \
   CONFIG.DEST_IDS {M00_AXI:0x1c0} \
   CONFIG.CATEGORY {ps_cci} \
 ] [get_bd_intf_pins /NOC_0/S00_AXI]

  set_property -dict [ list \
   CONFIG.DATA_WIDTH {128} \
   CONFIG.CONNECTIONS {M00_AXI { read_bw {5} write_bw {5} read_avg_burst {4} write_avg_burst {4}} MC_1 { read_bw {5} write_bw {5} read_avg_burst {4} write_avg_burst {4}} } \
   CONFIG.DEST_IDS {M00_AXI:0x1c0} \
   CONFIG.CATEGORY {ps_cci} \
 ] [get_bd_intf_pins /NOC_0/S01_AXI]

  set_property -dict [ list \
   CONFIG.DATA_WIDTH {128} \
   CONFIG.CONNECTIONS {M00_AXI { read_bw {5} write_bw {5} read_avg_burst {4} write_avg_burst {4}} MC_2 { read_bw {5} write_bw {5} read_avg_burst {4} write_avg_burst {4}} } \
   CONFIG.DEST_IDS {M00_AXI:0x1c0} \
   CONFIG.CATEGORY {ps_cci} \
 ] [get_bd_intf_pins /NOC_0/S02_AXI]

  set_property -dict [ list \
   CONFIG.DATA_WIDTH {128} \
   CONFIG.CONNECTIONS {M00_AXI { read_bw {5} write_bw {5} read_avg_burst {4} write_avg_burst {4}} MC_3 { read_bw {5} write_bw {5} read_avg_burst {4} write_avg_burst {4}} } \
   CONFIG.DEST_IDS {M00_AXI:0x1c0} \
   CONFIG.CATEGORY {ps_cci} \
 ] [get_bd_intf_pins /NOC_0/S03_AXI]

  set_property -dict [ list \
   CONFIG.DATA_WIDTH {128} \
   CONFIG.CONNECTIONS {MC_0 { read_bw {5} write_bw {5} read_avg_burst {4} write_avg_burst {4}} } \
   CONFIG.DEST_IDS {M00_AXI:0x200} \
   CONFIG.CATEGORY {ps_nci} \
 ] [get_bd_intf_pins /NOC_0/S04_AXI]

  set_property -dict [ list \
   CONFIG.DATA_WIDTH {128} \
   CONFIG.CONNECTIONS {MC_1 { read_bw {5} write_bw {5} read_avg_burst {4} write_avg_burst {4}} } \
   CONFIG.DEST_IDS {M00_AXI:0x200} \
   CONFIG.CATEGORY {ps_nci} \
 ] [get_bd_intf_pins /NOC_0/S05_AXI]

  set_property -dict [ list \
   CONFIG.DATA_WIDTH {128} \
   CONFIG.CONNECTIONS {MC_2 { read_bw {5} write_bw {5} read_avg_burst {4} write_avg_burst {4}} } \
   CONFIG.DEST_IDS {M00_AXI:0x200} \
   CONFIG.CATEGORY {ps_rpu} \
 ] [get_bd_intf_pins /NOC_0/S06_AXI]

  set_property -dict [ list \
   CONFIG.DATA_WIDTH {128} \
   CONFIG.CONNECTIONS {M00_AXI { read_bw {5} write_bw {5} read_avg_burst {4} write_avg_burst {4}} MC_3 { read_bw {5} write_bw {5} read_avg_burst {4} write_avg_burst {4}} } \
   CONFIG.DEST_IDS {M00_AXI:0x1c0} \
   CONFIG.CATEGORY {ps_pmc} \
 ] [get_bd_intf_pins /NOC_0/S07_AXI]

  set_property -dict [ list \
   CONFIG.DATA_WIDTH {256} \
   CONFIG.R_TRAFFIC_CLASS {LOW_LATENCY} \
   CONFIG.W_TRAFFIC_CLASS {BEST_EFFORT} \
   CONFIG.PHYSICAL_LOC {NOC_NMU512_X0Y0} \
   CONFIG.CONNECTIONS {MC_0 { read_bw {1782} write_bw {5} read_avg_burst {4} write_avg_burst {4}} } \
   CONFIG.CATEGORY {pl} \
 ] [get_bd_intf_pins /NOC_0/S08_AXI]

  set_property -dict [ list \
   CONFIG.DATA_WIDTH {256} \
   CONFIG.R_TRAFFIC_CLASS {LOW_LATENCY} \
   CONFIG.W_TRAFFIC_CLASS {BEST_EFFORT} \
   CONFIG.PHYSICAL_LOC {NOC_NMU512_X0Y1} \
   CONFIG.CONNECTIONS {MC_1 { read_bw {1782} write_bw {5} read_avg_burst {4} write_avg_burst {4}} } \
   CONFIG.CATEGORY {pl} \
 ] [get_bd_intf_pins /NOC_0/S09_AXI]

  set_property -dict [ list \
   CONFIG.DATA_WIDTH {256} \
   CONFIG.R_TRAFFIC_CLASS {LOW_LATENCY} \
   CONFIG.PHYSICAL_LOC {NOC_NMU512_X0Y2} \
   CONFIG.CONNECTIONS {MC_2 { read_bw {1782} write_bw {5} read_avg_burst {4} write_avg_burst {4}} } \
   CONFIG.CATEGORY {pl} \
 ] [get_bd_intf_pins /NOC_0/S10_AXI]

  set_property -dict [ list \
   CONFIG.DATA_WIDTH {32} \
   CONFIG.R_TRAFFIC_CLASS {BEST_EFFORT} \
   CONFIG.PHYSICAL_LOC {NOC_NMU512_X0Y3} \
   CONFIG.CONNECTIONS {MC_3 { read_bw {1782} write_bw {1782} read_avg_burst {4} write_avg_burst {4}} } \
   CONFIG.CATEGORY {pl} \
 ] [get_bd_intf_pins /NOC_0/S11_AXI]

  set_property -dict [ list \
   CONFIG.DATA_WIDTH {32} \
   CONFIG.CONNECTIONS {MC_0 { read_bw {1782} write_bw {1782} read_avg_burst {4} write_avg_burst {4}} } \
   CONFIG.CATEGORY {pl} \
 ] [get_bd_intf_pins /NOC_0/S12_AXI]

  set_property -dict [ list \
   CONFIG.DATA_WIDTH {256} \
   CONFIG.CONNECTIONS {MC_1 { read_bw {5} write_bw {1782} read_avg_burst {4} write_avg_burst {4}} } \
   CONFIG.CATEGORY {pl} \
 ] [get_bd_intf_pins /NOC_0/S13_AXI]

  set_property -dict [ list \
   CONFIG.DATA_WIDTH {32} \
   CONFIG.CONNECTIONS {MC_2 { read_bw {5} write_bw {5} read_avg_burst {4} write_avg_burst {4}} } \
   CONFIG.CATEGORY {pl} \
 ] [get_bd_intf_pins /NOC_0/S14_AXI]

  set_property -dict [ list \
   CONFIG.ASSOCIATED_BUSIF {S00_AXI} \
 ] [get_bd_pins /NOC_0/aclk0]

  set_property -dict [ list \
   CONFIG.ASSOCIATED_BUSIF {S01_AXI} \
 ] [get_bd_pins /NOC_0/aclk1]

  set_property -dict [ list \
   CONFIG.ASSOCIATED_BUSIF {S02_AXI} \
 ] [get_bd_pins /NOC_0/aclk2]

  set_property -dict [ list \
   CONFIG.ASSOCIATED_BUSIF {S03_AXI} \
 ] [get_bd_pins /NOC_0/aclk3]

  set_property -dict [ list \
   CONFIG.ASSOCIATED_BUSIF {S04_AXI} \
 ] [get_bd_pins /NOC_0/aclk4]

  set_property -dict [ list \
   CONFIG.ASSOCIATED_BUSIF {S05_AXI} \
 ] [get_bd_pins /NOC_0/aclk5]

  set_property -dict [ list \
   CONFIG.ASSOCIATED_BUSIF {S06_AXI} \
 ] [get_bd_pins /NOC_0/aclk6]

  set_property -dict [ list \
   CONFIG.ASSOCIATED_BUSIF {S07_AXI} \
 ] [get_bd_pins /NOC_0/aclk7]

  set_property -dict [ list \
   CONFIG.ASSOCIATED_BUSIF {S08_AXI:S09_AXI:S10_AXI:S11_AXI:S12_AXI} \
 ] [get_bd_pins /NOC_0/aclk8]

  set_property -dict [ list \
   CONFIG.ASSOCIATED_BUSIF {S13_AXI} \
 ] [get_bd_pins /NOC_0/aclk9]

  set_property -dict [ list \
   CONFIG.ASSOCIATED_BUSIF {M00_AXI} \
 ] [get_bd_pins /NOC_0/aclk10]

  set_property -dict [ list \
   CONFIG.ASSOCIATED_BUSIF {S14_AXI} \
 ] [get_bd_pins /NOC_0/aclk11]

  # Create instance: ai_engine_0, and set properties
  set ai_engine_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:ai_engine:2.0 ai_engine_0 ]
  set_property -dict [ list \
   CONFIG.CLK_NAMES {} \
   CONFIG.FIFO_TYPE_MI_AXIS {} \
   CONFIG.FIFO_TYPE_SI_AXIS {} \
   CONFIG.NAME_MI_AXIS {} \
   CONFIG.NAME_SI_AXIS {} \
   CONFIG.NUM_CLKS {0} \
   CONFIG.NUM_MI_AXI {0} \
   CONFIG.NUM_MI_AXIS {0} \
   CONFIG.NUM_SI_AXIS {0} \
 ] $ai_engine_0

  set_property -dict [ list \
   CONFIG.CATEGORY {NOC} \
 ] [get_bd_intf_pins /ai_engine_0/S00_AXI]

  # Create instance: audio_pipe
  create_hier_cell_audio_pipe [current_bd_instance .] audio_pipe

  # Create instance: axi_intc_0, and set properties
  set axi_intc_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_intc:4.1 axi_intc_0 ]

  # Create instance: axi_perf_mon_0, and set properties
  set axi_perf_mon_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_perf_mon:5.0 axi_perf_mon_0 ]
  set_property -dict [ list \
   CONFIG.C_ENABLE_ADVANCED {1} \
   CONFIG.C_ENABLE_EVENT_COUNT {1} \
   CONFIG.C_ENABLE_PROFILE {0} \
   CONFIG.C_NUM_MONITOR_SLOTS {6} \
   CONFIG.C_NUM_OF_COUNTERS {10} \
   CONFIG.ENABLE_EXT_TRIGGERS {0} \
 ] $axi_perf_mon_0

  # Create instance: axi_vip_m0, and set properties
  set axi_vip_m0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_vip:1.1 axi_vip_m0 ]
  set_property -dict [ list \
   CONFIG.ADDR_WIDTH {32} \
   CONFIG.ARUSER_WIDTH {0} \
   CONFIG.AWUSER_WIDTH {0} \
   CONFIG.BUSER_WIDTH {0} \
   CONFIG.DATA_WIDTH {32} \
   CONFIG.HAS_BRESP {1} \
   CONFIG.HAS_BURST {1} \
   CONFIG.HAS_CACHE {1} \
   CONFIG.HAS_LOCK {1} \
   CONFIG.HAS_PROT {1} \
   CONFIG.HAS_QOS {1} \
   CONFIG.HAS_REGION {1} \
   CONFIG.HAS_RRESP {1} \
   CONFIG.HAS_WSTRB {1} \
   CONFIG.ID_WIDTH {0} \
   CONFIG.INTERFACE_MODE {MASTER} \
   CONFIG.PROTOCOL {AXI4} \
   CONFIG.READ_WRITE_MODE {READ_WRITE} \
   CONFIG.RUSER_BITS_PER_BYTE {0} \
   CONFIG.RUSER_WIDTH {0} \
   CONFIG.SUPPORTS_NARROW {1} \
   CONFIG.WUSER_BITS_PER_BYTE {0} \
   CONFIG.WUSER_WIDTH {0} \
 ] $axi_vip_m0

  # Create instance: axi_vip_m1, and set properties
  set axi_vip_m1 [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_vip:1.1 axi_vip_m1 ]
  set_property -dict [ list \
   CONFIG.ADDR_WIDTH {32} \
   CONFIG.ARUSER_WIDTH {0} \
   CONFIG.AWUSER_WIDTH {0} \
   CONFIG.BUSER_WIDTH {0} \
   CONFIG.DATA_WIDTH {32} \
   CONFIG.HAS_BRESP {1} \
   CONFIG.HAS_BURST {1} \
   CONFIG.HAS_CACHE {1} \
   CONFIG.HAS_LOCK {1} \
   CONFIG.HAS_PROT {1} \
   CONFIG.HAS_QOS {1} \
   CONFIG.HAS_REGION {1} \
   CONFIG.HAS_RRESP {1} \
   CONFIG.HAS_WSTRB {1} \
   CONFIG.ID_WIDTH {0} \
   CONFIG.INTERFACE_MODE {MASTER} \
   CONFIG.PROTOCOL {AXI4} \
   CONFIG.READ_WRITE_MODE {READ_WRITE} \
   CONFIG.RUSER_BITS_PER_BYTE {0} \
   CONFIG.RUSER_WIDTH {0} \
   CONFIG.SUPPORTS_NARROW {1} \
   CONFIG.WUSER_BITS_PER_BYTE {0} \
   CONFIG.WUSER_WIDTH {0} \
 ] $axi_vip_m1

  # Create instance: axi_vip_s0, and set properties
  set axi_vip_s0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_vip:1.1 axi_vip_s0 ]
  set_property -dict [ list \
   CONFIG.INTERFACE_MODE {SLAVE} \
 ] $axi_vip_s0

  # Create instance: clk_wiz, and set properties
  set clk_wiz [ create_bd_cell -type ip -vlnv xilinx.com:ip:clk_wizard:1.0 clk_wiz ]
  set_property -dict [ list \
   CONFIG.CLKFBOUT_MULT {90.000000} \
   CONFIG.CLKOUT1_DIVIDE {20.000000} \
   CONFIG.CLKOUT2_DIVIDE {30.000000} \
   CONFIG.CLKOUT3_DIVIDE {15.000000} \
   CONFIG.CLKOUT4_DIVIDE {12} \
   CONFIG.CLKOUT_DRIVES {BUFG,BUFG,BUFG,BUFG,BUFG,BUFG,BUFG} \
   CONFIG.CLKOUT_DYN_PS {None,None,None,None,None,None,None} \
   CONFIG.CLKOUT_GROUPING {Auto,Auto,Auto,Auto,Auto,Auto,Auto} \
   CONFIG.CLKOUT_MATCHED_ROUTING {false,false,false,false,false,false,false} \
   CONFIG.CLKOUT_MBUFGCE_MODE {PERFORMANCE,PERFORMANCE,PERFORMANCE,PERFORMANCE,PERFORMANCE,PERFORMANCE,PERFORMANCE} \
   CONFIG.CLKOUT_PORT {clk_out_150,clk_out_100,clk_out_200,clk_out4,clk_out5,clk_out6,clk_out7} \
   CONFIG.CLKOUT_REQUESTED_DUTY_CYCLE {50.000,50.000,50.000,50.000,50.000,50.000,50.000} \
   CONFIG.CLKOUT_REQUESTED_OUT_FREQUENCY {150.000,100.000,200.000,150.000,100.000,100.000,100.000} \
   CONFIG.CLKOUT_REQUESTED_PHASE {0.000,0.000,0.000,0.000,0.000,0.000,0.000} \
   CONFIG.CLKOUT_USED {true,true,true,false,false,false,false} \
   CONFIG.DIVCLK_DIVIDE {3} \
   CONFIG.JITTER_SEL {Min_O_Jitter} \
   CONFIG.PRIM_SOURCE {Global_buffer} \
   CONFIG.RESET_TYPE {ACTIVE_LOW} \
   CONFIG.SECONDARY_IN_FREQ {100.000} \
   CONFIG.USE_LOCKED {true} \
   CONFIG.USE_PHASE_ALIGNMENT {true} \
   CONFIG.USE_RESET {true} \
 ] $clk_wiz

  # Create instance: clk_wiz_accel, and set properties
  set clk_wiz_accel [ create_bd_cell -type ip -vlnv xilinx.com:ip:clk_wizard:1.0 clk_wiz_accel ]
  set_property -dict [ list \
   CONFIG.BANDWIDTH {LOW} \
   CONFIG.CLKFBOUT_MULT {299.703125} \
   CONFIG.CLKOUT1_DIVIDE {10.000000} \
   CONFIG.CLKOUT2_DIVIDE {5.000000} \
   CONFIG.CLKOUT3_DIVIDE {12} \
   CONFIG.CLKOUT_DRIVES {BUFG,BUFG,BUFG,BUFG,BUFG,BUFG,BUFG} \
   CONFIG.CLKOUT_DYN_PS {None,None,None,None,None,None,None} \
   CONFIG.CLKOUT_GROUPING {Auto,Auto,Auto,Auto,Auto,Auto,Auto} \
   CONFIG.CLKOUT_MATCHED_ROUTING {false,false,false,false,false,false,false} \
   CONFIG.CLKOUT_PORT {clk_out_333,clk_out_666,clk_out3,clk_out4,clk_out5,clk_out6,clk_out7} \
   CONFIG.CLKOUT_REQUESTED_DUTY_CYCLE {50.000,50.000,50.000,50.000,50.000,50.000,50.000} \
   CONFIG.CLKOUT_REQUESTED_OUT_FREQUENCY {333,666,200,100.000,100.000,100.000,100.000} \
   CONFIG.CLKOUT_REQUESTED_PHASE {0.000,0.000,0.000,0.000,0.000,0.000,0.000} \
   CONFIG.CLKOUT_USED {true,true,false,false,false,false,false} \
   CONFIG.DIVCLK_DIVIDE {9} \
   CONFIG.PRIM_SOURCE {Global_buffer} \
   CONFIG.RESET_TYPE {ACTIVE_LOW} \
   CONFIG.USE_RESET {true} \
 ] $clk_wiz_accel

  # Create instance: display_pipe
  create_hier_cell_display_pipe [current_bd_instance .] display_pipe

  # Create instance: mipi_capture_pipe
  create_hier_cell_mipi_capture_pipe [current_bd_instance .] mipi_capture_pipe

  # Create instance: rst_processor_150MHz, and set properties
  set rst_processor_150MHz [ create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 rst_processor_150MHz ]
  set_property -dict [ list \
   CONFIG.C_NUM_PERP_ARESETN {1} \
 ] $rst_processor_150MHz

  # Create instance: rst_processor_pl_100Mhz, and set properties
  set rst_processor_pl_100Mhz [ create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 rst_processor_pl_100Mhz ]

  # Create instance: rst_processor_pl_200Mhz, and set properties
  set rst_processor_pl_200Mhz [ create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 rst_processor_pl_200Mhz ]

  # Create instance: rst_processor_pl_333Mhz, and set properties
  set rst_processor_pl_333Mhz [ create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 rst_processor_pl_333Mhz ]

  # Create instance: rst_processor_pl_666Mhz, and set properties
  set rst_processor_pl_666Mhz [ create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 rst_processor_pl_666Mhz ]

  # Create instance: smartconnect_accel0, and set properties
  set smartconnect_accel0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect:1.0 smartconnect_accel0 ]
  set_property -dict [ list \
   CONFIG.NUM_SI {1} \
 ] $smartconnect_accel0

  # Create instance: smartconnect_accel1, and set properties
  set smartconnect_accel1 [ create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect:1.0 smartconnect_accel1 ]
  set_property -dict [ list \
   CONFIG.NUM_SI {1} \
 ] $smartconnect_accel1

  # Create instance: smartconnect_gp0, and set properties
  set smartconnect_gp0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect:1.0 smartconnect_gp0 ]
  set_property -dict [ list \
   CONFIG.NUM_CLKS {3} \
   CONFIG.NUM_MI {14} \
   CONFIG.NUM_SI {1} \
 ] $smartconnect_gp0

  # Create instance: smartconnect_gp2, and set properties
  set smartconnect_gp2 [ create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect:1.0 smartconnect_gp2 ]
  set_property -dict [ list \
   CONFIG.NUM_SI {1} \
 ] $smartconnect_gp2

  # Create interface connections
  connect_bd_intf_net -intf_net CIPS_0_IF_PMC_NOC_AXI_0 [get_bd_intf_pins CIPS_0/PMC_NOC_AXI_0] [get_bd_intf_pins NOC_0/S07_AXI]
  connect_bd_intf_net -intf_net CIPS_0_IF_PS_NOC_CCI_0 [get_bd_intf_pins CIPS_0/FPD_CCI_NOC_0] [get_bd_intf_pins NOC_0/S00_AXI]
  connect_bd_intf_net -intf_net CIPS_0_IF_PS_NOC_CCI_1 [get_bd_intf_pins CIPS_0/FPD_CCI_NOC_1] [get_bd_intf_pins NOC_0/S01_AXI]
  connect_bd_intf_net -intf_net CIPS_0_IF_PS_NOC_CCI_2 [get_bd_intf_pins CIPS_0/FPD_CCI_NOC_2] [get_bd_intf_pins NOC_0/S02_AXI]
  connect_bd_intf_net -intf_net CIPS_0_IF_PS_NOC_CCI_3 [get_bd_intf_pins CIPS_0/FPD_CCI_NOC_3] [get_bd_intf_pins NOC_0/S03_AXI]
  connect_bd_intf_net -intf_net CIPS_0_IF_PS_NOC_NCI_0 [get_bd_intf_pins CIPS_0/FPD_AXI_NOC_0] [get_bd_intf_pins NOC_0/S04_AXI]
  connect_bd_intf_net -intf_net CIPS_0_IF_PS_NOC_NCI_1 [get_bd_intf_pins CIPS_0/FPD_AXI_NOC_1] [get_bd_intf_pins NOC_0/S05_AXI]
  connect_bd_intf_net -intf_net CIPS_0_IF_PS_NOC_RPU_0 [get_bd_intf_pins CIPS_0/NOC_LPD_AXI_0] [get_bd_intf_pins NOC_0/S06_AXI]
  connect_bd_intf_net -intf_net CIPS_0_M_AXI_GP2 [get_bd_intf_pins CIPS_0/M_AXI_LPD] [get_bd_intf_pins smartconnect_gp2/S00_AXI]
  connect_bd_intf_net -intf_net CLK_IN_D_0_1 [get_bd_intf_ports CLK_IN_AUDIO] [get_bd_intf_pins audio_pipe/CLK_IN_AUDIO]
  connect_bd_intf_net -intf_net NOC_0_CH0_LPDDR4_0 [get_bd_intf_ports CH0_LPDDR4_0_0] [get_bd_intf_pins NOC_0/CH0_LPDDR4_0]
  connect_bd_intf_net -intf_net NOC_0_CH0_LPDDR4_1 [get_bd_intf_ports CH0_LPDDR4_1_0] [get_bd_intf_pins NOC_0/CH0_LPDDR4_1]
  connect_bd_intf_net -intf_net NOC_0_CH1_LPDDR4_0 [get_bd_intf_ports CH1_LPDDR4_0_0] [get_bd_intf_pins NOC_0/CH1_LPDDR4_0]
  connect_bd_intf_net -intf_net NOC_0_CH1_LPDDR4_1 [get_bd_intf_ports CH1_LPDDR4_1_0] [get_bd_intf_pins NOC_0/CH1_LPDDR4_1]
  connect_bd_intf_net -intf_net NOC_0_M00_AXI [get_bd_intf_pins NOC_0/M00_AXI] [get_bd_intf_pins ai_engine_0/S00_AXI]
  connect_bd_intf_net -intf_net audio_pipe_m_axi_mm2s [get_bd_intf_pins NOC_0/S14_AXI] [get_bd_intf_pins audio_pipe/m_axi_mm2s]
  connect_bd_intf_net -intf_net audio_pipe_m_axis_mm2s [get_bd_intf_pins audio_pipe/m_axis_mm2s] [get_bd_intf_pins display_pipe/AUDIO_IN]
  connect_bd_intf_net -intf_net axi_1 [get_bd_intf_pins audio_pipe/s_axi_ctrl_acr] [get_bd_intf_pins smartconnect_gp0/M12_AXI]
  connect_bd_intf_net -intf_net axi_vip_1_M_AXI [get_bd_intf_pins axi_vip_m0/M_AXI] [get_bd_intf_pins smartconnect_accel0/S00_AXI]
  connect_bd_intf_net -intf_net axi_vip_2_M_AXI [get_bd_intf_pins axi_vip_m1/M_AXI] [get_bd_intf_pins smartconnect_accel1/S00_AXI]
  connect_bd_intf_net -intf_net cap_pipe_M00_AXI [get_bd_intf_pins NOC_0/S09_AXI] [get_bd_intf_pins display_pipe/vmix_mm_axi_vid_rd_1]
connect_bd_intf_net -intf_net [get_bd_intf_nets cap_pipe_M00_AXI] [get_bd_intf_pins NOC_0/S09_AXI] [get_bd_intf_pins axi_perf_mon_0/SLOT_1_AXI]
  connect_bd_intf_net -intf_net csi_mipi_phy_if_1 [get_bd_intf_ports csi_mipi_phy_if] [get_bd_intf_pins mipi_capture_pipe/csi_mipi_phy_if]
  connect_bd_intf_net -intf_net csirxss_s_axi_1 [get_bd_intf_pins mipi_capture_pipe/csirxss_s_axi] [get_bd_intf_pins smartconnect_gp0/M05_AXI]
  connect_bd_intf_net -intf_net display_pipe_HDMI_CTL_IIC [get_bd_intf_ports HDMI_CTL_IIC] [get_bd_intf_pins display_pipe/HDMI_CTL_IIC]
  connect_bd_intf_net -intf_net display_pipe_TX_DDC_OUT [get_bd_intf_ports TX_DDC_OUT] [get_bd_intf_pins display_pipe/TX_DDC_OUT]
  connect_bd_intf_net -intf_net display_pipe_m_axi_mm_video9 [get_bd_intf_pins NOC_0/S11_AXI] [get_bd_intf_pins smartconnect_accel0/M00_AXI]
connect_bd_intf_net -intf_net [get_bd_intf_nets display_pipe_m_axi_mm_video9] [get_bd_intf_pins axi_perf_mon_0/SLOT_3_AXI] [get_bd_intf_pins smartconnect_accel0/M00_AXI]
  connect_bd_intf_net -intf_net display_pipe_vmix_mm_axi_vid_rd_0 [get_bd_intf_pins NOC_0/S08_AXI] [get_bd_intf_pins display_pipe/vmix_mm_axi_vid_rd_0]
connect_bd_intf_net -intf_net [get_bd_intf_nets display_pipe_vmix_mm_axi_vid_rd_0] [get_bd_intf_pins NOC_0/S08_AXI] [get_bd_intf_pins axi_perf_mon_0/SLOT_0_AXI]
  connect_bd_intf_net -intf_net display_pipe_vmix_mm_axi_vid_rd_1 [get_bd_intf_pins NOC_0/S10_AXI] [get_bd_intf_pins display_pipe/vmix_mm_axi_vid_rd_2]
connect_bd_intf_net -intf_net [get_bd_intf_nets display_pipe_vmix_mm_axi_vid_rd_1] [get_bd_intf_pins NOC_0/S10_AXI] [get_bd_intf_pins axi_perf_mon_0/SLOT_2_AXI]
  connect_bd_intf_net -intf_net mipi_capture_pipe_M00_AXI [get_bd_intf_pins NOC_0/S13_AXI] [get_bd_intf_pins mipi_capture_pipe/M00_AXI]
connect_bd_intf_net -intf_net [get_bd_intf_nets mipi_capture_pipe_M00_AXI] [get_bd_intf_pins axi_perf_mon_0/SLOT_5_AXI] [get_bd_intf_pins mipi_capture_pipe/M00_AXI]
  connect_bd_intf_net -intf_net mipi_csi_rx_ss_IIC_sensor [get_bd_intf_ports sensor_iic] [get_bd_intf_pins mipi_capture_pipe/sensor_iic]
  connect_bd_intf_net -intf_net s_axi_CTRL1_1 [get_bd_intf_pins mipi_capture_pipe/s_axi_CTRL1] [get_bd_intf_pins smartconnect_gp0/M07_AXI]
  connect_bd_intf_net -intf_net s_axi_CTRL_1 [get_bd_intf_pins mipi_capture_pipe/s_axi_CTRL] [get_bd_intf_pins smartconnect_gp0/M06_AXI]
  connect_bd_intf_net -intf_net s_axi_ctrl_1_1 [get_bd_intf_pins mipi_capture_pipe/s_axi_ctrl_1] [get_bd_intf_pins smartconnect_gp0/M10_AXI]
  connect_bd_intf_net -intf_net s_axi_ctrl_vmix_1 [get_bd_intf_pins display_pipe/s_axi_ctrl_vmix] [get_bd_intf_pins smartconnect_gp0/M03_AXI]
  connect_bd_intf_net -intf_net smartconnect_accel_M00_AXI [get_bd_intf_pins NOC_0/S12_AXI] [get_bd_intf_pins smartconnect_accel1/M00_AXI]
connect_bd_intf_net -intf_net [get_bd_intf_nets smartconnect_accel_M00_AXI] [get_bd_intf_pins axi_perf_mon_0/SLOT_4_AXI] [get_bd_intf_pins smartconnect_accel1/M00_AXI]
  connect_bd_intf_net -intf_net smartconnect_gp0_M00_AXI [get_bd_intf_pins display_pipe/S_AXI] [get_bd_intf_pins smartconnect_gp0/M00_AXI]
  connect_bd_intf_net -intf_net smartconnect_gp0_M01_AXI [get_bd_intf_pins display_pipe/S_AXI_CPU_IN] [get_bd_intf_pins smartconnect_gp0/M01_AXI]
  connect_bd_intf_net -intf_net smartconnect_gp0_M02_AXI [get_bd_intf_pins display_pipe/axi4lite] [get_bd_intf_pins smartconnect_gp0/M02_AXI]
  connect_bd_intf_net -intf_net smartconnect_gp0_M04_AXI [get_bd_intf_pins mipi_capture_pipe/S_AXI] [get_bd_intf_pins smartconnect_gp0/M04_AXI]
  connect_bd_intf_net -intf_net smartconnect_gp0_M08_AXI [get_bd_intf_pins axi_intc_0/s_axi] [get_bd_intf_pins smartconnect_gp0/M08_AXI]
  connect_bd_intf_net -intf_net smartconnect_gp0_M11_AXI [get_bd_intf_pins axi_perf_mon_0/S_AXI] [get_bd_intf_pins smartconnect_gp0/M11_AXI]
  connect_bd_intf_net -intf_net smartconnect_gp0_M13_AXI [get_bd_intf_pins audio_pipe/s_axi_ctrl_aud_for] [get_bd_intf_pins smartconnect_gp0/M13_AXI]
  connect_bd_intf_net -intf_net smartconnect_gp2_M00_AXI [get_bd_intf_pins axi_vip_s0/S_AXI] [get_bd_intf_pins smartconnect_gp2/M00_AXI]
  connect_bd_intf_net -intf_net sys_clk0_0_1 [get_bd_intf_ports sys_clk0_0] [get_bd_intf_pins NOC_0/sys_clk0]
  connect_bd_intf_net -intf_net sys_clk1_0_1 [get_bd_intf_ports sys_clk1_0] [get_bd_intf_pins NOC_0/sys_clk1]
  connect_bd_intf_net -intf_net versal_cips_1_M_AXI_GP0 [get_bd_intf_pins CIPS_0/M_AXI_FPD] [get_bd_intf_pins smartconnect_gp0/S00_AXI]

  # Create port connections
  connect_bd_net -net CIPS_0_pl_resetn0 [get_bd_pins CIPS_0/pl0_resetn] [get_bd_pins audio_pipe/ext_reset_in] [get_bd_pins clk_wiz/resetn] [get_bd_pins clk_wiz_accel/resetn] [get_bd_pins rst_processor_150MHz/ext_reset_in] [get_bd_pins rst_processor_pl_100Mhz/ext_reset_in] [get_bd_pins rst_processor_pl_200Mhz/ext_reset_in] [get_bd_pins rst_processor_pl_333Mhz/ext_reset_in] [get_bd_pins rst_processor_pl_666Mhz/ext_reset_in]
  connect_bd_net -net CIPS_0_ps_pmc_noc_axi0_clk [get_bd_pins CIPS_0/pmc_axi_noc_axi0_clk] [get_bd_pins NOC_0/aclk7]
  connect_bd_net -net CIPS_0_ps_ps_noc_cci_axi0_clk [get_bd_pins CIPS_0/fpd_cci_noc_axi0_clk] [get_bd_pins NOC_0/aclk0]
  connect_bd_net -net CIPS_0_ps_ps_noc_cci_axi1_clk [get_bd_pins CIPS_0/fpd_cci_noc_axi1_clk] [get_bd_pins NOC_0/aclk1]
  connect_bd_net -net CIPS_0_ps_ps_noc_cci_axi2_clk [get_bd_pins CIPS_0/fpd_cci_noc_axi2_clk] [get_bd_pins NOC_0/aclk2]
  connect_bd_net -net CIPS_0_ps_ps_noc_cci_axi3_clk [get_bd_pins CIPS_0/fpd_cci_noc_axi3_clk] [get_bd_pins NOC_0/aclk3]
  connect_bd_net -net CIPS_0_ps_ps_noc_nci_axi0_clk [get_bd_pins CIPS_0/fpd_axi_noc_axi0_clk] [get_bd_pins NOC_0/aclk4]
  connect_bd_net -net CIPS_0_ps_ps_noc_nci_axi1_clk [get_bd_pins CIPS_0/fpd_axi_noc_axi1_clk] [get_bd_pins NOC_0/aclk5]
  connect_bd_net -net CIPS_0_ps_ps_noc_rpu_axi0_clk [get_bd_pins CIPS_0/lpd_axi_noc_clk] [get_bd_pins NOC_0/aclk6]
  connect_bd_net -net Din_1 [get_bd_pins CIPS_0/lpd_gpio_o] [get_bd_pins display_pipe/Din] [get_bd_pins mipi_capture_pipe/Din]
  connect_bd_net -net IDT_8T49N241_LOL_IN_1 [get_bd_ports IDT_8T49N241_LOL_IN] [get_bd_pins display_pipe/IDT_8T49N241_LOL_IN]
  connect_bd_net -net RX_DATA_IN_rxn_1 [get_bd_ports RX_DATA_IN_rxn] [get_bd_pins display_pipe/RX_DATA_IN_rxn]
  connect_bd_net -net RX_DATA_IN_rxp_1 [get_bd_ports RX_DATA_IN_rxp] [get_bd_pins display_pipe/RX_DATA_IN_rxp]
  connect_bd_net -net TX_HPD_IN_1 [get_bd_ports TX_HPD_IN] [get_bd_pins display_pipe/TX_HPD_IN]
  connect_bd_net -net TX_REFCLK_N_IN_1 [get_bd_ports TX_REFCLK_N_IN] [get_bd_pins display_pipe/TX_REFCLK_N_IN]
  connect_bd_net -net TX_REFCLK_P_IN_1 [get_bd_ports TX_REFCLK_P_IN] [get_bd_pins display_pipe/TX_REFCLK_P_IN]
  connect_bd_net -net ai_engine_0_s00_axi_aclk [get_bd_pins NOC_0/aclk10] [get_bd_pins ai_engine_0/s00_axi_aclk]
  connect_bd_net -net audio_pipe_BUFG_O [get_bd_pins NOC_0/aclk11] [get_bd_pins audio_pipe/aud_clk] [get_bd_pins display_pipe/s_axis_audio_aclk]
  connect_bd_net -net audio_pipe_aud_acr_cts_out [get_bd_pins audio_pipe/aud_acr_cts_out] [get_bd_pins display_pipe/acr_cts]
  connect_bd_net -net audio_pipe_aud_acr_n_out [get_bd_pins audio_pipe/aud_acr_n_out] [get_bd_pins display_pipe/acr_n]
  connect_bd_net -net audio_pipe_aud_acr_valid_out [get_bd_pins audio_pipe/aud_acr_valid_out] [get_bd_pins display_pipe/acr_valid]
  connect_bd_net -net audio_pipe_irq_mm2s [get_bd_pins CIPS_0/pl_ps_irq5] [get_bd_pins audio_pipe/irq_mm2s]
  connect_bd_net -net axi_intc_0_irq [get_bd_pins CIPS_0/pl_ps_irq10] [get_bd_pins axi_intc_0/irq]
  connect_bd_net -net axi_perf_mon_0_interrupt [get_bd_pins CIPS_0/pl_ps_irq4] [get_bd_pins axi_perf_mon_0/interrupt]
  connect_bd_net -net cap_pipe_Dout [get_bd_ports sensor_gpio_rst] [get_bd_pins mipi_capture_pipe/sensor_gpio_rst]
  connect_bd_net -net clk_wiz_accel_clk_out_333 [get_bd_pins clk_wiz_accel/clk_out_333] [get_bd_pins rst_processor_pl_333Mhz/slowest_sync_clk]
  connect_bd_net -net clk_wiz_accel_clk_out_666 [get_bd_pins clk_wiz_accel/clk_out_666] [get_bd_pins rst_processor_pl_666Mhz/slowest_sync_clk]
  connect_bd_net -net clk_wiz_clk_out2 [get_bd_pins CIPS_0/m_axi_fpd_aclk] [get_bd_pins audio_pipe/s_axi_lite_aclk] [get_bd_pins clk_wiz/clk_out_100] [get_bd_pins display_pipe/altclk] [get_bd_pins mipi_capture_pipe/s_axi_aclk] [get_bd_pins rst_processor_pl_100Mhz/slowest_sync_clk] [get_bd_pins smartconnect_gp0/aclk]
  connect_bd_net -net clk_wiz_clk_out3 [get_bd_pins NOC_0/aclk9] [get_bd_pins axi_perf_mon_0/core_aclk] [get_bd_pins axi_perf_mon_0/slot_5_axi_aclk] [get_bd_pins clk_wiz/clk_out_200] [get_bd_pins mipi_capture_pipe/video_clk] [get_bd_pins rst_processor_pl_200Mhz/slowest_sync_clk] [get_bd_pins smartconnect_gp0/aclk2]
  connect_bd_net -net display_pipe_TX_EN_OUT [get_bd_ports TX_EN_OUT] [get_bd_pins display_pipe/TX_EN_OUT]
  connect_bd_net -net display_pipe_iic2intc_irpt [get_bd_pins CIPS_0/pl_ps_irq3] [get_bd_pins display_pipe/iic2intc_irpt]
  connect_bd_net -net display_pipe_irq [get_bd_pins CIPS_0/pl_ps_irq0] [get_bd_pins display_pipe/irq]
  connect_bd_net -net display_pipe_irq1 [get_bd_pins CIPS_0/pl_ps_irq1] [get_bd_pins display_pipe/irq1]
  connect_bd_net -net display_pipe_txn_0 [get_bd_ports TX_DATA_OUT_txn] [get_bd_pins display_pipe/TX_DATA_OUT_txn]
  connect_bd_net -net display_pipe_txp_0 [get_bd_ports TX_DATA_OUT_txp] [get_bd_pins display_pipe/TX_DATA_OUT_txp]
  connect_bd_net -net display_pipe_vmix_intr [get_bd_pins CIPS_0/pl_ps_irq2] [get_bd_pins display_pipe/vmix_intr]
  connect_bd_net -net hdmi_clk_1 [get_bd_pins audio_pipe/hdmi_clk] [get_bd_pins display_pipe/tx_tmds_clk]
  connect_bd_net -net mipi_capture_pipe_csirxss_csi_irq [get_bd_pins CIPS_0/pl_ps_irq6] [get_bd_pins mipi_capture_pipe/csirxss_csi_irq]
  connect_bd_net -net mipi_capture_pipe_frm_buf_irq [get_bd_pins CIPS_0/pl_ps_irq8] [get_bd_pins mipi_capture_pipe/frm_buf_irq]
  connect_bd_net -net mipi_capture_pipe_iic2intc_irpt [get_bd_pins CIPS_0/pl_ps_irq7] [get_bd_pins mipi_capture_pipe/iic2intc_irpt]
  connect_bd_net -net mipi_csi2_rx_dout1 [get_bd_ports sensor_gpio_flash] [get_bd_ports sensor_gpio_spi_cs_n] [get_bd_pins mipi_capture_pipe/sensor_gpio_spi_cs_n]
  connect_bd_net -net net_clk_wiz_locked [get_bd_pins clk_wiz/locked] [get_bd_pins rst_processor_150MHz/dcm_locked] [get_bd_pins rst_processor_pl_100Mhz/dcm_locked] [get_bd_pins rst_processor_pl_200Mhz/dcm_locked]
  connect_bd_net -net net_mb_ss_0_clk_out2 [get_bd_pins CIPS_0/m_axi_lpd_aclk] [get_bd_pins NOC_0/aclk8] [get_bd_pins axi_intc_0/s_axi_aclk] [get_bd_pins axi_perf_mon_0/s_axi_aclk] [get_bd_pins axi_perf_mon_0/slot_0_axi_aclk] [get_bd_pins axi_perf_mon_0/slot_1_axi_aclk] [get_bd_pins axi_perf_mon_0/slot_2_axi_aclk] [get_bd_pins axi_perf_mon_0/slot_3_axi_aclk] [get_bd_pins axi_perf_mon_0/slot_4_axi_aclk] [get_bd_pins axi_vip_m0/aclk] [get_bd_pins axi_vip_m1/aclk] [get_bd_pins axi_vip_s0/aclk] [get_bd_pins clk_wiz/clk_out_150] [get_bd_pins display_pipe/s_axis_aclk] [get_bd_pins rst_processor_150MHz/slowest_sync_clk] [get_bd_pins smartconnect_accel0/aclk] [get_bd_pins smartconnect_accel1/aclk] [get_bd_pins smartconnect_gp0/aclk1] [get_bd_pins smartconnect_gp2/aclk]
  connect_bd_net -net net_mb_ss_0_dcm_locked [get_bd_pins axi_intc_0/s_axi_aresetn] [get_bd_pins display_pipe/aresetn1] [get_bd_pins rst_processor_150MHz/peripheral_aresetn]
  connect_bd_net -net net_mb_ss_0_s_axi_aclk [get_bd_pins CIPS_0/pl0_ref_clk] [get_bd_pins clk_wiz/clk_in1] [get_bd_pins clk_wiz_accel/clk_in1]
  connect_bd_net -net rst_processor_1_100M_peripheral_aresetn [get_bd_pins audio_pipe/s_axi_lite_aresetn] [get_bd_pins display_pipe/ARESETN] [get_bd_pins mipi_capture_pipe/s_axi_aresetn] [get_bd_pins rst_processor_pl_100Mhz/peripheral_aresetn]
  connect_bd_net -net rst_processor_1_150M_interconnect_aresetn [get_bd_pins axi_perf_mon_0/s_axi_aresetn] [get_bd_pins axi_perf_mon_0/slot_0_axi_aresetn] [get_bd_pins axi_perf_mon_0/slot_1_axi_aresetn] [get_bd_pins axi_perf_mon_0/slot_2_axi_aresetn] [get_bd_pins axi_perf_mon_0/slot_3_axi_aresetn] [get_bd_pins axi_perf_mon_0/slot_4_axi_aresetn] [get_bd_pins axi_vip_m0/aresetn] [get_bd_pins axi_vip_m1/aresetn] [get_bd_pins axi_vip_s0/aresetn] [get_bd_pins display_pipe/sc_aresetn] [get_bd_pins rst_processor_150MHz/interconnect_aresetn] [get_bd_pins smartconnect_accel0/aresetn] [get_bd_pins smartconnect_accel1/aresetn] [get_bd_pins smartconnect_gp2/aresetn]
  connect_bd_net -net rst_processor_pl_200Mhz_peripheral_aresetn [get_bd_pins axi_perf_mon_0/core_aresetn] [get_bd_pins axi_perf_mon_0/slot_5_axi_aresetn] [get_bd_pins mipi_capture_pipe/video_rst_n] [get_bd_pins rst_processor_pl_200Mhz/peripheral_aresetn]
  connect_bd_net -net s_axis_audio_aresetn_1 [get_bd_pins audio_pipe/aud_resetn_out] [get_bd_pins display_pipe/s_axis_audio_aresetn]
  connect_bd_net -net versal_cips_ss_interconnect_aresetn [get_bd_pins rst_processor_pl_100Mhz/interconnect_aresetn] [get_bd_pins smartconnect_gp0/aresetn]

  # Create address segments
  assign_bd_address -offset 0xA40C0000 -range 0x00010000 -target_address_space [get_bd_addr_spaces CIPS_0/Data1] [get_bd_addr_segs mipi_capture_pipe/cap_pipe/ISPPipeline_accel_0/s_axi_CTRL/Reg] -force
  assign_bd_address -offset 0x050000000000 -range 0x000180000000 -target_address_space [get_bd_addr_spaces CIPS_0/DATA_NCI0] [get_bd_addr_segs NOC_0/S04_AXI/C0_DDR_CH1x2] -force
  assign_bd_address -offset 0x050000000000 -range 0x000180000000 -target_address_space [get_bd_addr_spaces CIPS_0/DATA_CCI0] [get_bd_addr_segs NOC_0/S00_AXI/C0_DDR_CH1x2] -force
  assign_bd_address -offset 0x00000000 -range 0x80000000 -target_address_space [get_bd_addr_spaces CIPS_0/DATA_NCI0] [get_bd_addr_segs NOC_0/S04_AXI/C0_DDR_LOW0x2] -force
  assign_bd_address -offset 0x00000000 -range 0x80000000 -target_address_space [get_bd_addr_spaces CIPS_0/DATA_CCI0] [get_bd_addr_segs NOC_0/S00_AXI/C0_DDR_LOW0x2] -force
  assign_bd_address -offset 0x050000000000 -range 0x000180000000 -target_address_space [get_bd_addr_spaces CIPS_0/DATA_NCI1] [get_bd_addr_segs NOC_0/S05_AXI/C1_DDR_CH1x2] -force
  assign_bd_address -offset 0x050000000000 -range 0x000180000000 -target_address_space [get_bd_addr_spaces CIPS_0/DATA_CCI1] [get_bd_addr_segs NOC_0/S01_AXI/C1_DDR_CH1x2] -force
  assign_bd_address -offset 0x00000000 -range 0x80000000 -target_address_space [get_bd_addr_spaces CIPS_0/DATA_NCI1] [get_bd_addr_segs NOC_0/S05_AXI/C1_DDR_LOW0x2] -force
  assign_bd_address -offset 0x00000000 -range 0x80000000 -target_address_space [get_bd_addr_spaces CIPS_0/DATA_CCI1] [get_bd_addr_segs NOC_0/S01_AXI/C1_DDR_LOW0x2] -force
  assign_bd_address -offset 0x050000000000 -range 0x000180000000 -target_address_space [get_bd_addr_spaces CIPS_0/DATA_RPU0] [get_bd_addr_segs NOC_0/S06_AXI/C2_DDR_CH1x2] -force
  assign_bd_address -offset 0x050000000000 -range 0x000180000000 -target_address_space [get_bd_addr_spaces CIPS_0/DATA_CCI2] [get_bd_addr_segs NOC_0/S02_AXI/C2_DDR_CH1x2] -force
  assign_bd_address -offset 0x00000000 -range 0x80000000 -target_address_space [get_bd_addr_spaces CIPS_0/DATA_RPU0] [get_bd_addr_segs NOC_0/S06_AXI/C2_DDR_LOW0x2] -force
  assign_bd_address -offset 0x00000000 -range 0x80000000 -target_address_space [get_bd_addr_spaces CIPS_0/DATA_CCI2] [get_bd_addr_segs NOC_0/S02_AXI/C2_DDR_LOW0x2] -force
  assign_bd_address -offset 0x050000000000 -range 0x000180000000 -target_address_space [get_bd_addr_spaces CIPS_0/DATA_PMC] [get_bd_addr_segs NOC_0/S07_AXI/C3_DDR_CH1x2] -force
  assign_bd_address -offset 0x050000000000 -range 0x000180000000 -target_address_space [get_bd_addr_spaces CIPS_0/DATA_CCI3] [get_bd_addr_segs NOC_0/S03_AXI/C3_DDR_CH1x2] -force
  assign_bd_address -offset 0x00000000 -range 0x80000000 -target_address_space [get_bd_addr_spaces CIPS_0/DATA_PMC] [get_bd_addr_segs NOC_0/S07_AXI/C3_DDR_LOW0x2] -force
  assign_bd_address -offset 0x00000000 -range 0x80000000 -target_address_space [get_bd_addr_spaces CIPS_0/DATA_CCI3] [get_bd_addr_segs NOC_0/S03_AXI/C3_DDR_LOW0x2] -force
  assign_bd_address -offset 0x020000000000 -range 0x000100000000 -target_address_space [get_bd_addr_spaces CIPS_0/DATA_PMC] [get_bd_addr_segs ai_engine_0/S00_AXI/AIE_ARRAY_0] -force
  assign_bd_address -offset 0x020000000000 -range 0x000100000000 -target_address_space [get_bd_addr_spaces CIPS_0/DATA_CCI0] [get_bd_addr_segs ai_engine_0/S00_AXI/AIE_ARRAY_0] -force
  assign_bd_address -offset 0x020000000000 -range 0x000100000000 -target_address_space [get_bd_addr_spaces CIPS_0/DATA_CCI3] [get_bd_addr_segs ai_engine_0/S00_AXI/AIE_ARRAY_0] -force
  assign_bd_address -offset 0x020000000000 -range 0x000100000000 -target_address_space [get_bd_addr_spaces CIPS_0/DATA_CCI1] [get_bd_addr_segs ai_engine_0/S00_AXI/AIE_ARRAY_0] -force
  assign_bd_address -offset 0x020000000000 -range 0x000100000000 -target_address_space [get_bd_addr_spaces CIPS_0/DATA_CCI2] [get_bd_addr_segs ai_engine_0/S00_AXI/AIE_ARRAY_0] -force
  assign_bd_address -offset 0xA4200000 -range 0x00010000 -target_address_space [get_bd_addr_spaces CIPS_0/Data1] [get_bd_addr_segs audio_pipe/audio_formatter_0/s_axi_lite/reg0] -force
  assign_bd_address -offset 0xA4070000 -range 0x00010000 -target_address_space [get_bd_addr_spaces CIPS_0/Data1] [get_bd_addr_segs mipi_capture_pipe/mipi_csi_rx_ss/axi_iic_1_sensor/S_AXI/Reg] -force
  assign_bd_address -offset 0xA5000000 -range 0x00010000 -target_address_space [get_bd_addr_spaces CIPS_0/Data1] [get_bd_addr_segs axi_intc_0/S_AXI/Reg] -force
  assign_bd_address -offset 0xA4050000 -range 0x00010000 -target_address_space [get_bd_addr_spaces CIPS_0/Data1] [get_bd_addr_segs axi_perf_mon_0/S_AXI/Reg] -force
  assign_bd_address -offset 0x90000000 -range 0x00010000 -target_address_space [get_bd_addr_spaces CIPS_0/Data1] [get_bd_addr_segs axi_vip_s0/S_AXI/Reg] -force
  assign_bd_address -offset 0xA4010000 -range 0x00010000 -target_address_space [get_bd_addr_spaces CIPS_0/Data1] [get_bd_addr_segs display_pipe/hdmi_tx_phy/fmch_axi_iic/S_AXI/Reg] -force
  assign_bd_address -offset 0xA4210000 -range 0x00010000 -target_address_space [get_bd_addr_spaces CIPS_0/Data1] [get_bd_addr_segs audio_pipe/hdmi_acr_ctrl_0/axi/reg0] -force
  assign_bd_address -offset 0xA4000000 -range 0x00010000 -target_address_space [get_bd_addr_spaces CIPS_0/Data1] [get_bd_addr_segs display_pipe/hdmi_tx_phy/hdmi_gt_controller_1/axi4lite/Reg] -force
  assign_bd_address -offset 0xA4060000 -range 0x00010000 -target_address_space [get_bd_addr_spaces CIPS_0/Data1] [get_bd_addr_segs mipi_capture_pipe/mipi_csi_rx_ss/mipi_csi2_rx_subsyst_0/csirxss_s_axi/Reg] -force
  assign_bd_address -offset 0xA40D0000 -range 0x00010000 -target_address_space [get_bd_addr_spaces CIPS_0/Data1] [get_bd_addr_segs mipi_capture_pipe/cap_pipe/v_frmbuf_wr_0/s_axi_CTRL/Reg] -force
  assign_bd_address -offset 0xA4020000 -range 0x00020000 -target_address_space [get_bd_addr_spaces CIPS_0/Data1] [get_bd_addr_segs display_pipe/v_hdmi_tx_ss_0/S_AXI_CPU_IN/Reg] -force
  assign_bd_address -offset 0xA4040000 -range 0x00010000 -target_address_space [get_bd_addr_spaces CIPS_0/Data1] [get_bd_addr_segs display_pipe/v_mix_0/s_axi_CTRL/Reg] -force
  assign_bd_address -offset 0xA4080000 -range 0x00040000 -target_address_space [get_bd_addr_spaces CIPS_0/Data1] [get_bd_addr_segs mipi_capture_pipe/cap_pipe/v_proc_ss_0/s_axi_ctrl/Reg] -force
  assign_bd_address -offset 0x00000000 -range 0x80000000 -target_address_space [get_bd_addr_spaces axi_vip_m0/Master_AXI] [get_bd_addr_segs NOC_0/S11_AXI/C3_DDR_LOW0x2] -force
  assign_bd_address -offset 0x00000000 -range 0x80000000 -target_address_space [get_bd_addr_spaces axi_vip_m1/Master_AXI] [get_bd_addr_segs NOC_0/S12_AXI/C0_DDR_LOW0x2] -force
  assign_bd_address -offset 0x050000000000 -range 0x000180000000 -target_address_space [get_bd_addr_spaces audio_pipe/audio_formatter_0/m_axi_mm2s] [get_bd_addr_segs NOC_0/S14_AXI/C2_DDR_CH1x2] -force
  assign_bd_address -offset 0x00000000 -range 0x80000000 -target_address_space [get_bd_addr_spaces audio_pipe/audio_formatter_0/m_axi_mm2s] [get_bd_addr_segs NOC_0/S14_AXI/C2_DDR_LOW0x2] -force
  assign_bd_address -offset 0x050000000000 -range 0x000180000000 -target_address_space [get_bd_addr_spaces display_pipe/v_mix_0/Data_m_axi_mm_video1] [get_bd_addr_segs NOC_0/S08_AXI/C0_DDR_CH1x2] -force
  assign_bd_address -offset 0x050000000000 -range 0x000180000000 -target_address_space [get_bd_addr_spaces display_pipe/v_mix_0/Data_m_axi_mm_video2] [get_bd_addr_segs NOC_0/S08_AXI/C0_DDR_CH1x2] -force
  assign_bd_address -offset 0x050000000000 -range 0x000180000000 -target_address_space [get_bd_addr_spaces display_pipe/v_mix_0/Data_m_axi_mm_video6] [get_bd_addr_segs NOC_0/S08_AXI/C0_DDR_CH1x2] -force
  assign_bd_address -offset 0x050000000000 -range 0x000180000000 -target_address_space [get_bd_addr_spaces display_pipe/v_mix_0/Data_m_axi_mm_video5] [get_bd_addr_segs NOC_0/S08_AXI/C0_DDR_CH1x2] -force
  assign_bd_address -offset 0x00000000 -range 0x80000000 -target_address_space [get_bd_addr_spaces display_pipe/v_mix_0/Data_m_axi_mm_video1] [get_bd_addr_segs NOC_0/S08_AXI/C0_DDR_LOW0x2] -force
  assign_bd_address -offset 0x00000000 -range 0x80000000 -target_address_space [get_bd_addr_spaces display_pipe/v_mix_0/Data_m_axi_mm_video6] [get_bd_addr_segs NOC_0/S08_AXI/C0_DDR_LOW0x2] -force
  assign_bd_address -offset 0x00000000 -range 0x80000000 -target_address_space [get_bd_addr_spaces display_pipe/v_mix_0/Data_m_axi_mm_video5] [get_bd_addr_segs NOC_0/S08_AXI/C0_DDR_LOW0x2] -force
  assign_bd_address -offset 0x00000000 -range 0x80000000 -target_address_space [get_bd_addr_spaces display_pipe/v_mix_0/Data_m_axi_mm_video2] [get_bd_addr_segs NOC_0/S08_AXI/C0_DDR_LOW0x2] -force
  assign_bd_address -offset 0x050000000000 -range 0x000180000000 -target_address_space [get_bd_addr_spaces display_pipe/v_mix_0/Data_m_axi_mm_video4] [get_bd_addr_segs NOC_0/S09_AXI/C1_DDR_CH1x2] -force
  assign_bd_address -offset 0x050000000000 -range 0x000180000000 -target_address_space [get_bd_addr_spaces display_pipe/v_mix_0/Data_m_axi_mm_video3] [get_bd_addr_segs NOC_0/S09_AXI/C1_DDR_CH1x2] -force
  assign_bd_address -offset 0x050000000000 -range 0x000180000000 -target_address_space [get_bd_addr_spaces display_pipe/v_mix_0/Data_m_axi_mm_video7] [get_bd_addr_segs NOC_0/S09_AXI/C1_DDR_CH1x2] -force
  assign_bd_address -offset 0x050000000000 -range 0x000180000000 -target_address_space [get_bd_addr_spaces display_pipe/v_mix_0/Data_m_axi_mm_video8] [get_bd_addr_segs NOC_0/S09_AXI/C1_DDR_CH1x2] -force
  assign_bd_address -offset 0x00000000 -range 0x80000000 -target_address_space [get_bd_addr_spaces display_pipe/v_mix_0/Data_m_axi_mm_video4] [get_bd_addr_segs NOC_0/S09_AXI/C1_DDR_LOW0x2] -force
  assign_bd_address -offset 0x00000000 -range 0x80000000 -target_address_space [get_bd_addr_spaces display_pipe/v_mix_0/Data_m_axi_mm_video3] [get_bd_addr_segs NOC_0/S09_AXI/C1_DDR_LOW0x2] -force
  assign_bd_address -offset 0x00000000 -range 0x80000000 -target_address_space [get_bd_addr_spaces display_pipe/v_mix_0/Data_m_axi_mm_video7] [get_bd_addr_segs NOC_0/S09_AXI/C1_DDR_LOW0x2] -force
  assign_bd_address -offset 0x00000000 -range 0x80000000 -target_address_space [get_bd_addr_spaces display_pipe/v_mix_0/Data_m_axi_mm_video8] [get_bd_addr_segs NOC_0/S09_AXI/C1_DDR_LOW0x2] -force
  assign_bd_address -offset 0x050000000000 -range 0x000180000000 -target_address_space [get_bd_addr_spaces display_pipe/v_mix_0/Data_m_axi_mm_video9] [get_bd_addr_segs NOC_0/S10_AXI/C2_DDR_CH1x2] -force
  assign_bd_address -offset 0x00000000 -range 0x80000000 -target_address_space [get_bd_addr_spaces display_pipe/v_mix_0/Data_m_axi_mm_video9] [get_bd_addr_segs NOC_0/S10_AXI/C2_DDR_LOW0x2] -force
  assign_bd_address -offset 0x050000000000 -range 0x000180000000 -target_address_space [get_bd_addr_spaces mipi_capture_pipe/cap_pipe/v_frmbuf_wr_0/Data_m_axi_mm_video] [get_bd_addr_segs NOC_0/S13_AXI/C1_DDR_CH1x2] -force
  assign_bd_address -offset 0x00000000 -range 0x80000000 -target_address_space [get_bd_addr_spaces mipi_capture_pipe/cap_pipe/v_frmbuf_wr_0/Data_m_axi_mm_video] [get_bd_addr_segs NOC_0/S13_AXI/C1_DDR_LOW0x2] -force


  # Restore current instance
  current_bd_instance $oldCurInst

  # Create PFM attributes
  set_property PFM_NAME {xilinx.com:xd:vck190_base_trd_platform1:1.0} [get_files [current_bd_design].bd]
  set_property PFM.AXI_PORT {M00_AXI {memport "NOC_MASTER"} S15_AXI {memport "MIG" sptag "NOC_accel" memory "NOC_0 C3_DDR_LOW0x2" is_range "true"} S16_AXI {memport "MIG" sptag "NOC_accel" memory "NOC_0 C0_DDR_LOW0x2" is_range "true"} S17_AXI {memport "MIG" sptag "NOC_accel" memory "NOC_0 C1_DDR_LOW0x2" is_range "true"} S18_AXI {memport "MIG" sptag "NOC_accel" memory "NOC_0 C2_DDR_LOW0x2" is_range "true"} S19_AXI {memport "MIG" sptag "NOC_accel" memory "NOC_0 C3_DDR_LOW0x2" is_range "true"} S20_AXI {memport "MIG" sptag "NOC_accel" memory "NOC_0 C0_DDR_LOW0x2" is_range "true"}} [get_bd_cells /NOC_0]
  set_property PFM.IRQ {intr { id 0 range 32 }} [get_bd_cells /axi_intc_0]
  set_property PFM.CLOCK {clk_out_150 {id "0" is_default "true" proc_sys_reset "/rst_processor_150MHz" status "fixed" freq_hz "149998499"} clk_out_100 {id "1" is_default "false" proc_sys_reset "/rst_processor_pl_100Mhz" status "fixed" freq_hz "99999000"} clk_out_200 {id "2" is_default "false" proc_sys_reset "/rst_processor_pl_200Mhz" status "fixed" freq_hz "199998000"}} [get_bd_cells /clk_wiz]
  set_property PFM.CLOCK {clk_out_333 {id "3" is_default "false" proc_sys_reset "/rst_processor_pl_333Mhz" status "fixed" freq_hz "333000000"} clk_out_666 {id "4" is_default "false" proc_sys_reset "/rst_processor_pl_666Mhz" status "fixed" freq_hz "666000000"}} [get_bd_cells /clk_wiz_accel]
  set_property PFM.AXI_PORT {S01_AXI {memport "MIG" sptag "NOC_pl" memory "NOC_0 C0_DDR_LOW0x2"} S02_AXI {memport "MIG" sptag "NOC_pl" memory "NOC_0 C0_DDR_LOW0x2"} S03_AXI {memport "MIG" sptag "NOC_pl" memory "NOC_0 C0_DDR_LOW0x2"} S04_AXI {memport "MIG" sptag "NOC_pl" memory "NOC_0 C0_DDR_LOW0x2"} S05_AXI {memport "MIG" sptag "NOC_pl" memory "NOC_0 C0_DDR_LOW0x2"} S06_AXI {memport "MIG" sptag "NOC_pl" memory "NOC_0 C0_DDR_LOW0x2"} S07_AXI {memport "MIG" sptag "NOC_pl" memory "NOC_0 C0_DDR_LOW0x2"} S08_AXI {memport "MIG" sptag "NOC_pl" memory "NOC_0 C0_DDR_LOW0x2"} S09_AXI {memport "MIG" sptag "NOC_pl" memory "NOC_0 C0_DDR_LOW0x2"} S10_AXI {memport "MIG" sptag "NOC_pl" memory "NOC_0 C0_DDR_LOW0x2"} S11_AXI {memport "MIG" sptag "NOC_pl" memory "NOC_0 C0_DDR_LOW0x2"} S12_AXI {memport "MIG" sptag "NOC_pl" memory "NOC_0 C0_DDR_LOW0x2"} S13_AXI {memport "MIG" sptag "NOC_pl" memory "NOC_0 C0_DDR_LOW0x2"} S14_AXI {memport "MIG" sptag "NOC_pl" memory "NOC_0 C0_DDR_LOW0x2"} S15_AXI {memport "MIG" sptag "NOC_pl" memory "NOC_0 C0_DDR_LOW0x2"}} [get_bd_cells /smartconnect_accel0]
  set_property PFM.AXI_PORT {S01_AXI {memport "MIG" sptag "NOC_aie" memory "NOC_0 C1_DDR_LOW0x2"} S02_AXI {memport "MIG" sptag "NOC_aie" memory "NOC_0 C1_DDR_LOW0x2"} S03_AXI {memport "MIG" sptag "NOC_aie" memory "NOC_0 C1_DDR_LOW0x2"} S04_AXI {memport "MIG" sptag "NOC_aie" memory "NOC_0 C1_DDR_LOW0x2"} S05_AXI {memport "MIG" sptag "NOC_aie" memory "NOC_0 C1_DDR_LOW0x2"} S06_AXI {memport "MIG" sptag "NOC_aie" memory "NOC_0 C1_DDR_LOW0x2"} S07_AXI {memport "MIG" sptag "NOC_aie" memory "NOC_0 C1_DDR_LOW0x2"} S08_AXI {memport "MIG" sptag "NOC_aie" memory "NOC_0 C1_DDR_LOW0x2"} S09_AXI {memport "MIG" sptag "NOC_aie" memory "NOC_0 C1_DDR_LOW0x2"} S10_AXI {memport "MIG" sptag "NOC_aie" memory "NOC_0 C1_DDR_LOW0x2"} S11_AXI {memport "MIG" sptag "NOC_aie" memory "NOC_0 C1_DDR_LOW0x2"} S12_AXI {memport "MIG" sptag "NOC_aie" memory "NOC_0 C1_DDR_LOW0x2"} S13_AXI {memport "MIG" sptag "NOC_aie" memory "NOC_0 C1_DDR_LOW0x2"} S14_AXI {memport "MIG" sptag "NOC_aie" memory "NOC_0 C1_DDR_LOW0x2"} S15_AXI {memport "MIG" sptag "NOC_aie" memory "NOC_0 C1_DDR_LOW0x2"}} [get_bd_cells /smartconnect_accel1]
  set_property PFM.AXI_PORT {M01_AXI {memport "M_AXI_GP" sptag "" memory ""} M02_AXI {memport "M_AXI_GP" sptag "" memory ""} M03_AXI {memport "M_AXI_GP" sptag "" memory ""} M04_AXI {memport "M_AXI_GP" sptag "" memory ""} M05_AXI {memport "M_AXI_GP" sptag "" memory ""} M06_AXI {memport "M_AXI_GP" sptag "" memory ""} M07_AXI {memport "M_AXI_GP" sptag "" memory ""} M08_AXI {memport "M_AXI_GP" sptag "" memory ""} M09_AXI {memport "M_AXI_GP" sptag "" memory ""} M10_AXI {memport "M_AXI_GP" sptag "" memory ""} M11_AXI {memport "M_AXI_GP" sptag "" memory ""} M12_AXI {memport "M_AXI_GP" sptag "" memory ""} M13_AXI {memport "M_AXI_GP" sptag "" memory ""} M14_AXI {memport "M_AXI_GP" sptag "" memory ""} M15_AXI {memport "M_AXI_GP" sptag "" memory ""}} [get_bd_cells /smartconnect_gp2]


  validate_bd_design
  save_bd_design
}
# End of create_root_design()


##################################################################
# MAIN FLOW
##################################################################

create_root_design ""


