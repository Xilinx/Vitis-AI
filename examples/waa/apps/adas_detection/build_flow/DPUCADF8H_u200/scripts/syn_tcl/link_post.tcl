#####################################
#
puts "Start to source [info script]"


variable loc [file normalize [info script]]
regexp {build_flow/DPUCADF8H_u200} $loc script_dir
set SDA_PATH "[string range $loc -1 [expr [string first $script_dir $loc] -1 ] ]build_flow/DPUCADF8H_u200"

source $SDA_PATH/scripts/user_setup/env_config.tcl
source $SDA_PATH/scripts/user_setup/user_setup.tcl
source $SDA_PATH/scripts/proc_tcl/proc_vivado.tcl


if { $BUFFER_STRATEGY == "true" & $VIVADO_VER == "201901" } {
    source $SDA_PATH/scripts/constraints/property/config_mss.tcl
}
