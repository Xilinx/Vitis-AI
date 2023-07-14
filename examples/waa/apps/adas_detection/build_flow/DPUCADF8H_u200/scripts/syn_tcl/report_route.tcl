puts "Start to source [info script]"

set SDA_PATH  [pwd]

source $SDA_PATH/scripts/user_setup/env_config.tcl
source $SDA_PATH/scripts/user_setup/user_setup.tcl
source $SDA_PATH/scripts/proc_tcl/proc_vivado.tcl

if {$VIVADO_VER == "201802"} {
    set ROUTE_DCP $SDA_PATH/_x/link/vivado/prj/prj.runs/impl_1/pfm_top_wrapper_routed_error.dcp
} elseif {$VIVADO_VER == "201901"} {
    set ROUTE_DCP $SDA_PATH/_x/link/vivado/vpl/prj/prj.runs/impl_1/pfm_top_wrapper_routed_error.dcp
}

open_checkpoint $ROUTE_DCP

make_outputs "route"
