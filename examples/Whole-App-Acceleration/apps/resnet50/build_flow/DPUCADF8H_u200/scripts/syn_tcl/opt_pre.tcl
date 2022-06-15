#####################################
# Tcl file used in OPT_DESIGN.TCL.PRE

puts "Start to source [info script]"


#####################################

variable loc [file normalize [info script]]
regexp {build_flow/DPUCADF8H_u200} $loc script_dir
set SDA_PATH "[string range $loc -1 [expr [string first $script_dir $loc] -1 ] ]build_flow/DPUCADF8H_u200"

source $SDA_PATH/scripts/user_setup/env_config.tcl
source $SDA_PATH/scripts/user_setup/user_setup.tcl
source $SDA_PATH/scripts/proc_tcl/proc_vivado.tcl

make_outputs "synth"



if { (($SHELL_VER =="202002")||($SHELL_VER=="microsoft"))&&($BOARD == "u250" ) } {
# cross uram
    resize_pblock [get_pblocks  pblock_dynamic_SLR0] -add {SLICE_X38Y240:SLICE_X41Y299}
    resize_pblock [get_pblocks  pblock_dynamic_SLR0] -add {SLICE_X101Y240:SLICE_X104Y299}
# cross 7 Column laguna   
    resize_pblock [get_pblocks  pblock_dynamic_SLR0] -add {SLICE_X17Y240:SLICE_X21Y299}
    resize_pblock [get_pblocks  pblock_dynamic_SLR0] -add {SLICE_X35Y240:SLICE_X36Y299}
    resize_pblock [get_pblocks  pblock_dynamic_SLR0] -add {SLICE_X48Y240:SLICE_X52Y299}
    resize_pblock [get_pblocks  pblock_dynamic_SLR0] -add {SLICE_X61Y240:SLICE_X65Y299}
    resize_pblock [get_pblocks  pblock_dynamic_SLR0] -add {SLICE_X83Y240:SLICE_X87Y299}
    resize_pblock [get_pblocks  pblock_dynamic_SLR0] -add {SLICE_X95Y240:SLICE_X99Y299}
    resize_pblock [get_pblocks  pblock_dynamic_SLR0] -add {SLICE_X109Y240:SLICE_X112Y299}
#    resize_pblock [get_pblocks  pblock_dynamic_SLR0] -add {DSP48E2_X0Y96:DSP48E2_X12Y143}
#    resize_pblock [get_pblocks  pblock_dynamic_SLR0] -add {DSP48E2_X14Y96:DSP48E2_X15Y143}
#    resize_pblock [get_pblocks  pblock_dynamic_SLR0] -add {RAMB36_X0Y48:RAMB36_X7Y71}
    resize_pblock [get_pblocks  pblock_dynamic_SLR0] -add {URAM288_X0Y64:URAM288_X1Y79}
    resize_pblock [get_pblocks  pblock_dynamic_SLR0] -add {LAGUNA_X2Y120:LAGUNA_X15Y359}
    

# cross uram
    resize_pblock [get_pblocks  pblock_dynamic_SLR2] -add {SLICE_X38Y420:SLICE_X41Y479}
    resize_pblock [get_pblocks  pblock_dynamic_SLR2] -add {SLICE_X101Y420:SLICE_X104Y479}
# cross 7 Column laguna   
    resize_pblock [get_pblocks  pblock_dynamic_SLR2] -add {SLICE_X17Y420:SLICE_X21Y479}
    resize_pblock [get_pblocks  pblock_dynamic_SLR2] -add {SLICE_X35Y420:SLICE_X36Y479}
    resize_pblock [get_pblocks  pblock_dynamic_SLR2] -add {SLICE_X48Y420:SLICE_X52Y479}
    resize_pblock [get_pblocks  pblock_dynamic_SLR2] -add {SLICE_X61Y420:SLICE_X65Y479}
    resize_pblock [get_pblocks  pblock_dynamic_SLR2] -add {SLICE_X83Y420:SLICE_X87Y479}
    resize_pblock [get_pblocks  pblock_dynamic_SLR2] -add {SLICE_X95Y420:SLICE_X99Y479}
    resize_pblock [get_pblocks  pblock_dynamic_SLR2] -add {SLICE_X109Y420:SLICE_X112Y479}

#    resize_pblock [get_pblocks  pblock_dynamic_SLR2] -add {DSP48E2_X0Y144:DSP48E2_X12Y191}
#    resize_pblock [get_pblocks  pblock_dynamic_SLR2] -add {DSP48E2_X14Y144:DSP48E2_X15Y191}
#    resize_pblock [get_pblocks  pblock_dynamic_SLR2] -add {RAMB36_X0Y72:RAMB36_X7Y95}
    resize_pblock [get_pblocks  pblock_dynamic_SLR2] -add {URAM288_X0Y112:URAM288_X1Y127}
    resize_pblock [get_pblocks  pblock_dynamic_SLR2] -add {LAGUNA_X2Y360:LAGUNA_X15Y599}

    create_pblock slr0_top
    resize_pblock [get_pblocks  slr0_top] -add {SLICE_X17Y180:SLICE_X112Y239}
    resize_pblock [get_pblocks  slr0_top] -add {LAGUNA_X2Y120:LAGUNA_X15Y239}
    
    create_pblock slr1_bottom
#    resize_pblock [get_pblocks  slr1_bottom] -add {SLICE_X0Y300:SLICE_X50Y359}
# cross uram
    resize_pblock [get_pblocks  slr1_bottom] -add {SLICE_X38Y240:SLICE_X41Y299}
    resize_pblock [get_pblocks  slr1_bottom] -add {SLICE_X101Y240:SLICE_X104Y299}
# cross 7 Column laguna   
    resize_pblock [get_pblocks  slr1_bottom] -add {SLICE_X17Y240:SLICE_X21Y299}
    resize_pblock [get_pblocks  slr1_bottom] -add {SLICE_X35Y240:SLICE_X36Y299}
    resize_pblock [get_pblocks  slr1_bottom] -add {SLICE_X48Y240:SLICE_X52Y299}
    resize_pblock [get_pblocks  slr1_bottom] -add {SLICE_X61Y240:SLICE_X65Y299}
    resize_pblock [get_pblocks  slr1_bottom] -add {SLICE_X83Y240:SLICE_X87Y299}
    resize_pblock [get_pblocks  slr1_bottom] -add {SLICE_X95Y240:SLICE_X99Y299}
    resize_pblock [get_pblocks  slr1_bottom] -add {SLICE_X109Y240:SLICE_X112Y299}
    resize_pblock [get_pblocks  slr1_bottom] -add {LAGUNA_X2Y240:LAGUNA_X15Y359}
    
    create_pblock slr1_top
# cross uram
    resize_pblock [get_pblocks  slr1_top] -add {SLICE_X38Y419:SLICE_X41Y479}
    resize_pblock [get_pblocks  slr1_top] -add {SLICE_X101Y419:SLICE_X104Y479}
# cross 7 Column laguna   
    resize_pblock [get_pblocks  slr1_top] -add {SLICE_X17Y419:SLICE_X21Y479}
    resize_pblock [get_pblocks  slr1_top] -add {SLICE_X35Y419:SLICE_X36Y479}
    resize_pblock [get_pblocks  slr1_top] -add {SLICE_X48Y419:SLICE_X52Y479}
    resize_pblock [get_pblocks  slr1_top] -add {SLICE_X61Y419:SLICE_X65Y479}
    resize_pblock [get_pblocks  slr1_top] -add {SLICE_X83Y419:SLICE_X87Y479}
    resize_pblock [get_pblocks  slr1_top] -add {SLICE_X95Y419:SLICE_X99Y479}
    resize_pblock [get_pblocks  slr1_top] -add {SLICE_X109Y419:SLICE_X112Y479}
    resize_pblock [get_pblocks  slr1_top] -add {LAGUNA_X2Y360:LAGUNA_X15Y479}
    
    create_pblock slr2_bottom
    resize_pblock [get_pblocks  slr2_bottom] -add {SLICE_X17Y480:SLICE_X112Y539}
    resize_pblock [get_pblocks  slr2_bottom] -add {LAGUNA_X2Y480:LAGUNA_X15Y599}

}



#if { ($SHELL_VER =="202002")&&($BOARD == "u200" ) } {
#    resize_pblock [get_pblocks  pblock_dynamic_SLR0] -add {CLOCKREGION_X0Y0:CLOCKREGION_X5Y4}
#    resize_pblock [get_pblocks  pblock_dynamic_SLR0] -add {CLOCKREGION_X0Y5:CLOCKREGION_X1Y6}
#    resize_pblock [get_pblocks  pblock_dynamic_SLR0] -add {CLOCKREGION_X0Y7:CLOCKREGION_X2Y7}
#    resize_pblock [get_pblocks  pblock_dynamic_SLR2] -add {CLOCKREGION_X0Y8:CLOCKREGION_X2Y9}
#    resize_pblock [get_pblocks  pblock_dynamic_SLR2] -add {CLOCKREGION_X0Y10:CLOCKREGION_X5Y14}
#}

#if { ($SHELL_VER =="202002")&&($BOARD == "u250" ) } {
#    resize_pblock [get_pblocks  pblock_dynamic_SLR0] -add {CLOCKREGION_X0Y0:CLOCKREGION_X7Y3}
#    resize_pblock [get_pblocks  pblock_dynamic_SLR0] -add {CLOCKREGION_X0Y4:CLOCKREGION_X3Y5}
#    resize_pblock [get_pblocks  pblock_dynamic_SLR2] -add {CLOCKREGION_X0Y6:CLOCKREGION_X3Y7}
#    resize_pblock [get_pblocks  pblock_dynamic_SLR2] -add {CLOCKREGION_X0Y8:CLOCKREGION_X7Y11}
#    resize_pblock [get_pblocks  pblock_dynamic_SLR3] -add {CLOCKREGION_X0Y12:CLOCKREGION_X7Y15}
#}




