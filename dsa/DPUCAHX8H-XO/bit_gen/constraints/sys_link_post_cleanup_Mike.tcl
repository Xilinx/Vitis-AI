# (c) Copyright 2019 Xilinx, Inc. All rights reserved.
#
# This file contains confidential and proprietary information
# of Xilinx, Inc. and is protected under U.S. and
# international copyright and other intellectual property
# laws.
#
# DISCLAIMER
# This disclaimer is not a license and does not grant any
# rights to the materials distributed herewith. Except as
# otherwise provided in a valid license issued to you by
# Xilinx, and to the maximum extent permitted by applicable
# law: (1) THESE MATERIALS ARE MADE AVAILABLE "AS IS" AND
# WITH ALL FAULTS, AND XILINX HEREBY DISCLAIMS ALL WARRANTIES
# AND CONDITIONS, EXPRESS, IMPLIED, OR STATUTORY, INCLUDING
# BUT NOT LIMITED TO WARRANTIES OF MERCHANTABILITY, NON-
# INFRINGEMENT, OR FITNESS FOR ANY PARTICULAR PURPOSE; and
# (2) Xilinx shall not be liable (whether in contract or tort,
# including negligence, or under any other theory of
# liability) for any loss or damage of any kind or nature
# related to, arising under or in connection with these
# materials, including for any direct, or any indirect,
# special, incidental, or consequential loss or damage
# (including loss of data, profits, goodwill, or any type of
# loss or damage suffered as a result of any action brought
# by a third party) even if such damage or loss was
# reasonably foreseeable or Xilinx had been advised of the
# possibility of the same.
#
# CRITICAL APPLICATIONS
# Xilinx products are not designed or intended to be fail-
# safe, or for use in any application requiring fail-safe
# performance, such as life-support or safety devices or
# systems, Class III medical devices, nuclear facilities,
# applications related to the deployment of airbags, or any
# other applications that could lead to death, personal
# injury, or severe property or environmental damage
# (individually and collectively, "Critical
# Applications"). Customer assumes the sole risk and
# liability of any use of Xilinx products in Critical
# Applications, subject only to applicable laws and
# regulations governing limitations on product liability.
#
# THIS COPYRIGHT NOTICE AND DISCLAIMER MUST BE RETAINED AS
# PART OF THIS FILE AT ALL TIMES.
############################################################
#

# --------------------------------------------------------------
# Set up DSA-specific variables

# DDR/PLRAM memory_subsystem name + complete output
set __postsyslink_memory_subsystem_instance "memory_subsystem"
set __postsyslink_memory_subsystem_complete_port "sdx_mss_init_complete"
# Number of host ports on DDR/PLRAM MSS
set __postsyslink_mss_inst_num_host_ports 1

# SC between DDR/HBM
# Will get deleted if profiling is disabled + MSS is unused
set __postsyslink_smartconnect_instance "xdma_smartconnect"
# Number of ports on this instance
set __postsyslink_sc_inst_expected_ports 2

# HMSS segment naming when mapped - should be HBM_MEMXX but may not be
set __postsyslink_seg_prefix "hmss_mem"

# List of IPs to delete
# Will get deleted if MSS is unused(profiling or no-profiling)
set __postsyslink_ip_delete_list [list slr2/shutdown_slr2 slr2/axi_interconnect_0 memory_subsystem slr1/axi_cdc_xdma slr1/sdx_mss_regslice regslice_pipe_ctrl_mgntpf init_combine_mss init_cal_combine_mss]

# --------------------------------------------------------------
# Get the MSS instance and figure out what's connected
# Cannot remove when there are connections other than host (kernels/profiling etc)

# Get instance
if {[llength [get_bd_cells $__postsyslink_memory_subsystem_instance]] == 0} {
   send_msg_id {post_sys_link_tcl_cleanup 1-1} ERROR "Post Sys Link TCL routine cannot find the MSS instance :: $__postsyslink_memory_subsystem_instance"
} elseif {[llength [get_bd_cells $__postsyslink_memory_subsystem_instance]] == 1} {
   set mss_inst [get_bd_cells $__postsyslink_memory_subsystem_instance]
} else {
   send_msg_id {post_sys_link_tcl_cleanup 1-1} ERROR "Post Sys Link TCL routine round too many MSS instances :: [get_bd_cells $__postsyslink_memory_subsystem_instance]"
}

set __postsyslink_mss_in_use 1
# How many slave ports
if {[llength [get_property CONFIG.NUM_SI $mss_inst]] > 0} {
   set mss_inst_num_ports [get_property CONFIG.NUM_SI [get_bd_cells $mss_inst]]
   if {$mss_inst_num_ports == $__postsyslink_mss_inst_num_host_ports} {
      send_msg_id {post_sys_link_tcl_cleanup 1-1} INFO "Nothing other than host connected to $mss_inst - some cleanup will happen to free logic resources"
      set __postsyslink_mss_in_use 0
   } else {
   }
} else {
   send_msg_id {post_sys_link_tcl_cleanup 1-1} ERROR "NUM_SI CONFIG property not found on $mss_inst"
}

# --------------------------------------------------------------
# Only if the memory subsystem is not in use can we delete anything

if {$__postsyslink_mss_in_use == 0} {

   # --------------------------------------------------------------
   # Remove specified IPs - loosely
   foreach __postsyslink_ip $__postsyslink_ip_delete_list {
      if {[llength [get_bd_cells $__postsyslink_ip]] > 0} {
         send_msg_id {post_sys_link_tcl_cleanup 1-2} INFO "Deleting IP $__postsyslink_ip + associated nets as unused "

         if {[llength [get_bd_intf_pins -quiet -of [get_bd_cells $__postsyslink_ip]]] > 0} {
            foreach intf_pin [get_bd_intf_pins -of [get_bd_cells $__postsyslink_ip]] {
               if {[llength [get_bd_intf_nets -quiet -of [get_bd_intf_pins $intf_pin]]] > 0} {
                  disconnect_bd_intf_net [get_bd_intf_nets -of [get_bd_intf_pins $intf_pin]] [get_bd_intf_pins $intf_pin]
               }
	         }
         }

         if {[llength [get_bd_pins -of [get_bd_cells $__postsyslink_ip]]] > 0} {
            foreach pin [get_bd_pins -of [get_bd_cells $__postsyslink_ip]] {
               if {[llength [get_bd_intf_pin -quiet -of $pin]] == 0} {
                  if {[llength [get_bd_nets -quiet -of [get_bd_pins $pin]]] > 0} {
                     disconnect_bd_net [get_bd_nets -of [get_bd_pins $pin]] [get_bd_pins $pin]
                  }
               }
   	      }
         }

         delete_bd_objs [get_bd_cells $__postsyslink_ip]
         
      } else {
         send_msg_id {post_sys_link_tcl_cleanup 1-2} ERROR "IP $__postsyslink_ip not found for delete"
      }
   }
   
   # --------------------------------------------------------------
   # Tie off MSS Init done to ensure that XRT does not hang
   create_bd_cell -type ip -vlnv xilinx.com:ip:xlconstant:1.1 xlconstant_0
   if {[llength [get_bd_ports $__postsyslink_memory_subsystem_complete_port]] > 0} {
      connect_bd_net [get_bd_ports $__postsyslink_memory_subsystem_complete_port] [get_bd_pins xlconstant_0/dout]
   } else {
      send_msg_id {post_sys_link_tcl_cleanup 1-2} ERROR "Cannot find MSS complete port - $__postsyslink_memory_subsystem_complete_port. There is a requirement for this to be tied off."
   }

   # --------------------------------------------------------------
   # Get the SmartConnect instance and figure out what's connected
   # Profiling is enabled when there are more than expected ports
   
   # Get instance
   if {[llength [get_bd_cells $__postsyslink_smartconnect_instance]] == 0} {
      send_msg_id {post_sys_link_tcl_cleanup 1-3} ERROR "Post Sys Link TCL routine cannot find the SC instance :: $__postsyslink_smartconnect_instance"
   } elseif {[llength [get_bd_cells $__postsyslink_smartconnect_instance]] == 1} {
      set sc_inst [get_bd_cells $__postsyslink_smartconnect_instance]
   } else {
      send_msg_id {post_sys_link_tcl_cleanup 1-3} ERROR "Post Sys Link TCL routine found too many SC instances :: [get_bd_cells $__postsyslink_smartconnect_instance]"
   }

   set __postsyslink_profiling_enabled 1
   # How many master ports
   if {[llength [get_property CONFIG.NUM_MI $sc_inst]] > 0} {
      set num_ports_sc_inst [get_property CONFIG.NUM_MI $sc_inst]
      if {$num_ports_sc_inst > $__postsyslink_sc_inst_expected_ports} {
         send_msg_id {post_sys_link_tcl_cleanup 1-3} INFO "Profiling port is enabled on $sc_inst - $sc_inst will be preserved"
         set __postsyslink_profiling_enabled 1
      } elseif {$num_ports_sc_inst == $__postsyslink_sc_inst_expected_ports} {
         send_msg_id {post_sys_link_tcl_cleanup 1-3} INFO "Profiling port is not enabled on $sc_inst - $sc_inst will be removed"
         set __postsyslink_profiling_enabled 0
      } else {
         send_msg_id {post_sys_link_tcl_cleanup 1-3} ERROR "Unexpected number of ports on $sc_inst, expected $__postsyslink_sc_inst_expected_ports or above, got $num_ports_sc_inst"
      }
   } else {
      send_msg_id {post_sys_link_tcl_cleanup 1-3} ERROR "NUM_MI CONFIG property not found on $sc_inst"
   }

   # --------------------------------------------------------------
   # If profiling is enabled then leave the port hanging - safer
   # Otherwise delete the Smartconnect entirely
   # Assumption that HBM is connected to lowest port

   if {$__postsyslink_profiling_enabled == 0} {

      # Find the ports currently connected to slave and master port zero - assumption (also direction connection assumption
      # in case of hmss)
      set __postsyslink_sc_inst_slave_port [get_bd_intf_pins $sc_inst/S00_AXI]
      set __postsyslink_sc_inst_master_port [get_bd_intf_pins $sc_inst/M00_AXI]
      set __postsyslink_sc_inst_slave_port_to [find_bd_objs -thru_hier -relation connected_to $__postsyslink_sc_inst_slave_port]
      set __postsyslink_sc_inst_master_port_to [find_bd_objs -thru_hier -relation connected_to $__postsyslink_sc_inst_master_port]
      
      # Now find all the addressing segments and their offsets/ranges
      set __postsyslink_addr_segs        [get_bd_addr_segs -addressing -of $__postsyslink_sc_inst_master_port_to]
      set __postsyslink_addr_seg_sources [get_bd_addr_segs -addressable -of $__postsyslink_sc_inst_master_port_to]

      # Match source segs to destination segs and capture all of the RANGEs and OFFSET and NAMES
      # This is required as we want to remap to the same addresses after (normally controlled by HMSS)
      # HBM offsets are fixed
      set __postsyslink_addr_seg_dict [dict create]
      foreach __postsyslink_addressable_seg $__postsyslink_addr_seg_sources {
         set __postsyslink_key [get_property NAME $__postsyslink_addressable_seg]
         set __postsyslink_key_match [regsub "HBM_MEM" $__postsyslink_key $__postsyslink_seg_prefix]
         foreach __postsyslink_addressing_seg $__postsyslink_addr_segs {
            set dest_name [get_property NAME $__postsyslink_addressing_seg]
            # Names will match *HBM_MEMXX or *seg_prefix if never unmapped
            if {[regexp -- $__postsyslink_key_match $dest_name] | [regexp -- $__postsyslink_key $dest_name]} {
               dict set __postsyslink_addr_seg_dict $__postsyslink_key name       $dest_name
               dict set __postsyslink_addr_seg_dict $__postsyslink_key offset     [get_property OFFSET $__postsyslink_addressing_seg]
               dict set __postsyslink_addr_seg_dict $__postsyslink_key range      [get_property RANGE $__postsyslink_addressing_seg]
               dict set __postsyslink_addr_seg_dict $__postsyslink_key addr_space [get_bd_addr_spaces -of $__postsyslink_addressing_seg]
               dict set __postsyslink_addr_seg_dict $__postsyslink_key addr_seg   [get_bd_addr_segs $__postsyslink_sc_inst_master_port_to/$__postsyslink_key]
            }
         }
      }

      # Delete the Smartconnect
      delete_bd_objs $sc_inst
      
      # Connect the identified nets
      connect_bd_intf_net [get_bd_intf_pins $__postsyslink_sc_inst_slave_port_to] [get_bd_intf_pins $__postsyslink_sc_inst_master_port_to]
      
      # Remap all of the same segments to the same locations (don't want HBM segments to shift)
      foreach __postsyslink_key [dict keys $__postsyslink_addr_seg_dict] {
         set __postsyslink_name       [dict get $__postsyslink_addr_seg_dict $__postsyslink_key name]
         set __postsyslink_offset     [dict get $__postsyslink_addr_seg_dict $__postsyslink_key offset]
         set __postsyslink_range      [dict get $__postsyslink_addr_seg_dict $__postsyslink_key range]
         set __postsyslink_addr_space [dict get $__postsyslink_addr_seg_dict $__postsyslink_key addr_space]
         set __postsyslink_addr_seg   [dict get $__postsyslink_addr_seg_dict $__postsyslink_key addr_seg]
         create_bd_addr_seg -range $__postsyslink_range -offset $__postsyslink_offset [get_bd_addr_spaces $__postsyslink_addr_space] [get_bd_addr_segs $__postsyslink_addr_seg] $__postsyslink_name
      }

   }

}

# Validate design
# validate_bd_design
