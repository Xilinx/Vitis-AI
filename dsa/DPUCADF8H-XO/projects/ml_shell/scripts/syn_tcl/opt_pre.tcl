
puts "Start to source [info script]"



variable loc [file normalize [info script]]
regexp {projects/ml_shell} $loc script_dir
set SDA_PATH "[string range $loc -1 [expr [string first $script_dir $loc] -1 ] ]projects/ml_shell"

source $SDA_PATH/scripts/user_setup/env_config.tcl
source $SDA_PATH/scripts/user_setup/user_setup.tcl
source $SDA_PATH/scripts/proc_tcl/proc_vivado.tcl

make_outputs "synth"



if { ($PLATFORM =="xilinx_u250_gen3x16_xdma_2_1")||($PLATFORM=="xilinx_u250_gen3x16_xdma_3_1") } {
    for { set idx 0}  {$idx < $DPU_NUM} {incr idx} { 
        if { $DPU_SLR_IDX($idx) == 0 } {
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
         }
         if { $DPU_SLR_IDX($idx) == 2 } {
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
         }
      }
}






