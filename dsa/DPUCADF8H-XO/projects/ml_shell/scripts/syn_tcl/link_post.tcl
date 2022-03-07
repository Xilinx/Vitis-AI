puts "Start to source [info script]"


variable loc [file normalize [info script]]
regexp {projects/ml_shell} $loc script_dir
set SDA_PATH "[string range $loc -1 [expr [string first $script_dir $loc] -1 ] ]projects/ml_shell"

source $SDA_PATH/scripts/user_setup/env_config.tcl
source $SDA_PATH/scripts/user_setup/user_setup.tcl
source $SDA_PATH/scripts/proc_tcl/proc_vivado.tcl



