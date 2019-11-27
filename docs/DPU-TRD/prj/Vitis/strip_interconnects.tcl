set ps_cell [get_bd_cells * -hierarchical -quiet -filter {VLNV =~ "xilinx.com:ip:zynq_ultra_ps_e:*"}]
if {$ps_cell ne ""} {
	set pfm_ports_dict [get_property PFM.AXI_PORT $ps_cell]
	foreach {port} [get_bd_intf_pins $ps_cell/* -quiet -filter {MODE=~"Slave"}] {
		if {[dict exists $pfm_ports_dict [bd::utils::get_short_name $port]]} {
			set attached_interconnect [bd::utils::get_parent [find_bd_objs -relation connected_to $port ]]
			set intfCount [get_property CONFIG.NUM_SI $attached_interconnect]
			for {set i 0} {$i < $intfCount} {incr i} {
				set_property -dict [list CONFIG.S[format %02d $i]_HAS_REGSLICE {0} CONFIG.S[format %02d $i]_HAS_DATA_FIFO {0}] $attached_interconnect

			}
			set intfCount [get_property CONFIG.NUM_MI $attached_interconnect]
			for {set i 0} {$i < $intfCount} {incr i} {
				set_property -dict [list CONFIG.M[format %02d $i]_HAS_REGSLICE {0} CONFIG.M[format %02d $i]_HAS_DATA_FIFO {0}] $attached_interconnect
			}
		}
	}
}
set intr_cell [get_bd_cells * -hierarchical -quiet -filter {NAME =~ "interconnect_axifull"}]
if {$intr_cell ne ""} {
	set intfCount [get_property CONFIG.NUM_SI $intr_cell]
	for {set i 0} {$i < $intfCount} {incr i} {
		set_property -dict [list CONFIG.S[format %02d $i]_HAS_REGSLICE {0} CONFIG.S[format %02d $i]_HAS_DATA_FIFO {0}] $intr_cell
	}
	set intfCount [get_property CONFIG.NUM_MI $intr_cell]
	for {set i 0} {$i < $intfCount} {incr i} {
		set_property -dict [list CONFIG.M[format %02d $i]_HAS_REGSLICE {0} CONFIG.M[format %02d $i]_HAS_DATA_FIFO {0}] $intr_cell
	}
}


