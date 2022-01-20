set ip_delete_list [list memory_subsystem /SLR1/regslice_data /SLR1/axi_cdc_data /SLR1/axi_vip_data /SLR0/regslice_data /SLR0/axi_cdc_data /SLR0/axi_vip_data]
foreach ip [get_bd_cells $ip_delete_list] {
    #First, disconnect IP's connection
    foreach intf_pin [get_bd_intf_pins -of [get_bd_cells $ip]] {
	if { [get_bd_intf_nets -quiet -of [get_bd_intf_pins $intf_pin]] != "" } {
	    puts "Disconnecting intf net-[get_bd_intf_nets -of [get_bd_intf_pins $intf_pin]] and intf pin-$intf_pin";
	    disconnect_bd_intf_net [get_bd_intf_nets -of [get_bd_intf_pins $intf_pin]] $intf_pin
	} else {
	    puts "WARNING: Intf pin - $intf_pin is unconnected"
	}
    }
    foreach pin [get_bd_pins -of [get_bd_cells $ip] -filter TYPE!=undef] {
	if { [get_bd_nets -quiet -of [get_bd_pins $pin]] != "" } {
	    puts "Disconnecting net-[get_bd_nets -of [get_bd_pins $pin]] and pin-$pin";
	    disconnect_bd_net [get_bd_nets -of [get_bd_pins $pin]] $pin
	} else {
	    puts "WARNING: Pin - $pin is unconnected"
	}
    }

    #Secondly, remove IPs
    delete_bd_objs [get_bd_cells $ip]	
}

set __postsyslink_seg_prefix "hmss_mem"
set rmv_sc [list axi_data_sc]
foreach sc_inst [get_bd_cells $rmv_sc] {
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
puts "Debug -1: $__postsyslink_addressable_seg"
         set __postsyslink_key [get_property NAME $__postsyslink_addressable_seg]
         set __postsyslink_key_match [regsub "HBM_MEM" $__postsyslink_key $__postsyslink_seg_prefix]
puts "Debug 0: $__postsyslink_key and $__postsyslink_key_match"
         foreach __postsyslink_addressing_seg $__postsyslink_addr_segs {
            set dest_name [get_property NAME $__postsyslink_addressing_seg]
puts "Debug 1: $__postsyslink_addressing_seg and $dest_name"
            # Names will match *HBM_MEMXX or *seg_prefix if never unmapped
            if {[regexp -- $__postsyslink_key_match $dest_name] | [regexp -- $__postsyslink_key $dest_name]} {
               dict set __postsyslink_addr_seg_dict $__postsyslink_key name       $dest_name
               dict set __postsyslink_addr_seg_dict $__postsyslink_key offset     [get_property OFFSET $__postsyslink_addressing_seg]
               dict set __postsyslink_addr_seg_dict $__postsyslink_key range      [get_property RANGE $__postsyslink_addressing_seg]
               dict set __postsyslink_addr_seg_dict $__postsyslink_key addr_space [get_bd_addr_spaces -of $__postsyslink_addressing_seg]
               dict set __postsyslink_addr_seg_dict $__postsyslink_key addr_seg   [get_bd_addr_segs $__postsyslink_sc_inst_master_port_to/$__postsyslink_key]
            }
         }
puts "Debug 2: [dict get $__postsyslink_addr_seg_dict $__postsyslink_key name] [dict get $__postsyslink_addr_seg_dict $__postsyslink_key offset] [dict get $__postsyslink_addr_seg_dict $__postsyslink_key range] [dict get $__postsyslink_addr_seg_dict $__postsyslink_key addr_space] [dict get $__postsyslink_addr_seg_dict $__postsyslink_key addr_seg]"
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

#FIXME, there are some floating nets

if { 0 } {
for {set i 0} {$i < 200} {set i [expr $i+1]} {
    undo
}
}
