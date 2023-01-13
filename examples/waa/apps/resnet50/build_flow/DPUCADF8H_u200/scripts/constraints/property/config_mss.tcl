puts "Start to source [info script]"

##  use buffer strategy to config buffer size of smartconnect in memory-subsystem
set mss [get_bd_cells -filter {VLNV=~*sdx_memory_subsystem*}]
set mss_ap [get_property CONFIG.ADVANCED_PROPERTIES $mss]
dict set mss_ap buffer_strategy default_si_buffer_r_size 32
dict set mss_ap buffer_strategy default_si_buffer_w_size 32
set_property CONFIG.ADVANCED_PROPERTIES $mss_ap $mss
