#####################################
# Tcl file used in OPT_DESIGN.TCL.POST
#
puts "Start to source [info script]"

##################
# set_property or set_param tcl command is used in the following tcl scripts
source $SDA_PATH/scripts/constraints/property/vivado_property.tcl


# ##################
# # clock constrains are in the following constraints files
source $SDA_PATH/scripts/constraints/timing/mmcm.tcl
source $SDA_PATH/scripts/constraints/timing/clock_converter.tcl
source $SDA_PATH/scripts/constraints/timing/gen_clocks.tcl


# ##################
# # optimize Double Rate clock region with larger weight

#if { $BOARD == "u200" } {
#    if { $DPU_NUM==1 } {
#        group_path -name SLR${SLR}_ACLK_DR -weight 2
#    } else {
#        for { set idx 0}  {$idx < $DPU_NUM} {incr idx} { 
#            group_path -name SLR${idx}_ACLK_DR -weight 2
#        }
#    }
#}
if { $BOARD == "u200" } {
    if { $DPU_NUM==1 } {
        group_path -name SLR${SLR}_ACLK_DR -weight 2
    } 
    if { $DPU_NUM==2 } {
        group_path -name SLR0_ACLK_DR -weight 2
        group_path -name SLR2_ACLK_DR -weight 2
    }
}


##################
# design timing constrains are in the following constraints files

if { $DPU_NUM==1 } {
    set DPU_V3_TOP [expr \$SLR${SLR}_DPU_V3_TOP]
    puts "Start to source timing constraints for [expr \$SLR${SLR}_DPU_V3_TOP]"
    source $SDA_PATH/scripts/constraints/timing/jiangsha.tcl
    source $SDA_PATH/scripts/constraints/timing/yuqian.tcl
    source $SDA_PATH/scripts/constraints/timing/wangxi.tcl
    source $SDA_PATH/scripts/constraints/timing/junbin.tcl
} elseif { ($DPU_NUM==2 )&&($BOARD=="u200")} {
        set idx 0
        set DPU_V3_TOP [expr \$SLR${idx}_DPU_V3_TOP]
        puts "Start to source timing constraints for [expr \$SLR${idx}_DPU_V3_TOP]"
        source $SDA_PATH/scripts/constraints/timing/jiangsha.tcl
        source $SDA_PATH/scripts/constraints/timing/yuqian.tcl
        source $SDA_PATH/scripts/constraints/timing/wangxi.tcl
        source $SDA_PATH/scripts/constraints/timing/junbin.tcl
        set idx 2
        set DPU_V3_TOP [expr \$SLR${idx}_DPU_V3_TOP]
        puts "Start to source timing constraints for [expr \$SLR${idx}_DPU_V3_TOP]"
        source $SDA_PATH/scripts/constraints/timing/jiangsha.tcl
        source $SDA_PATH/scripts/constraints/timing/yuqian.tcl
        source $SDA_PATH/scripts/constraints/timing/wangxi.tcl
        source $SDA_PATH/scripts/constraints/timing/junbin.tcl
} elseif {($BOARD=="u250")&&(($SHELL_VER=="202002")||($SHELL_VER=="microsoft"))&&($DPU_NUM==2 )} {
        set DPU_V3_TOP [expr \$SLR0_DPU_V3_TOP]
        source $SDA_PATH/scripts/constraints/timing/jiangsha.tcl
        source $SDA_PATH/scripts/constraints/timing/yuqian.tcl
        source $SDA_PATH/scripts/constraints/timing/wangxi.tcl
        source $SDA_PATH/scripts/constraints/timing/junbin.tcl
        set DPU_V3_TOP [expr \$SLR2_DPU_V3_TOP]
        source $SDA_PATH/scripts/constraints/timing/jiangsha.tcl
        source $SDA_PATH/scripts/constraints/timing/yuqian.tcl
        source $SDA_PATH/scripts/constraints/timing/wangxi.tcl
        source $SDA_PATH/scripts/constraints/timing/junbin.tcl
} elseif {($BOARD=="u250")&&(($SHELL_VER=="202002")||($SHELL_VER=="microsoft"))&&($DPU_NUM==3 )} {
        set DPU_V3_TOP [expr \$SLR0_DPU_V3_TOP]
        source $SDA_PATH/scripts/constraints/timing/jiangsha.tcl
        source $SDA_PATH/scripts/constraints/timing/yuqian.tcl
        source $SDA_PATH/scripts/constraints/timing/wangxi.tcl
        source $SDA_PATH/scripts/constraints/timing/junbin.tcl
        set DPU_V3_TOP [expr \$SLR2_DPU_V3_TOP]
        source $SDA_PATH/scripts/constraints/timing/jiangsha.tcl
        source $SDA_PATH/scripts/constraints/timing/yuqian.tcl
        source $SDA_PATH/scripts/constraints/timing/wangxi.tcl
        source $SDA_PATH/scripts/constraints/timing/junbin.tcl
        set DPU_V3_TOP [expr \$SLR3_DPU_V3_TOP]
        source $SDA_PATH/scripts/constraints/timing/jiangsha.tcl
        source $SDA_PATH/scripts/constraints/timing/yuqian.tcl
        source $SDA_PATH/scripts/constraints/timing/wangxi.tcl
        source $SDA_PATH/scripts/constraints/timing/junbin.tcl
} else {
    for { set idx 0}  {$idx < $DPU_NUM} {incr idx} { 
        set DPU_V3_TOP [expr \$SLR${idx}_DPU_V3_TOP]
        puts "Start to source timing constraints for [expr \$SLR${idx}_DPU_V3_TOP]"
        source $SDA_PATH/scripts/constraints/timing/jiangsha.tcl
        source $SDA_PATH/scripts/constraints/timing/yuqian.tcl
        source $SDA_PATH/scripts/constraints/timing/wangxi.tcl
        source $SDA_PATH/scripts/constraints/timing/junbin.tcl
    }
}


##################
# physical constrains are in the following constraints files
if { ($SHELL_VER=="201803")||($SHELL_VER=="aws") } {
if { $DPU_NUM==1 } {
    set DPU_V3_TOP [expr \$SLR${SLR}_DPU_V3_TOP]
    puts "Start to source timing constraints for [expr \$SLR${SLR}_DPU_V3_TOP]"
    source $SDA_PATH/scripts/constraints/physical/$BOARD/slr${SLR}/bram_loc_wangxi.tcl
    source $SDA_PATH/scripts/constraints/physical/$BOARD/slr${SLR}/uram_loc_wangxi.tcl
    source $SDA_PATH/scripts/constraints/physical/$BOARD/slr${SLR}/dsp_loc_wangxi.tcl
    source $SDA_PATH/scripts/constraints/physical/$BOARD/slr${SLR}/mss.tcl
    if { $BOARD == "u200" } {
        source $SDA_PATH/scripts/constraints/physical/$BOARD/slr${SLR}/other_locs.tcl  
    }    
} elseif { ($DPU_NUM==2 )&&($BOARD=="u200")} {
        set idx 0
        set DPU_V3_TOP [expr \$SLR${idx}_DPU_V3_TOP]
        puts "Start to source timing constraints for [expr \$SLR${idx}_DPU_V3_TOP]"
        source $SDA_PATH/scripts/constraints/physical/$BOARD/slr${idx}/bram_loc_wangxi.tcl
        source $SDA_PATH/scripts/constraints/physical/$BOARD/slr${idx}/uram_loc_wangxi.tcl
        source $SDA_PATH/scripts/constraints/physical/$BOARD/slr${idx}/dsp_loc_wangxi.tcl
        source $SDA_PATH/scripts/constraints/physical/$BOARD/slr${idx}/mss.tcl
        source $SDA_PATH/scripts/constraints/physical/$BOARD/slr${idx}/other_locs.tcl
        set idx 2
        set DPU_V3_TOP [expr \$SLR${idx}_DPU_V3_TOP]
        puts "Start to source timing constraints for [expr \$SLR${idx}_DPU_V3_TOP]"
        source $SDA_PATH/scripts/constraints/physical/$BOARD/slr${idx}/bram_loc_wangxi.tcl
        source $SDA_PATH/scripts/constraints/physical/$BOARD/slr${idx}/uram_loc_wangxi.tcl
        source $SDA_PATH/scripts/constraints/physical/$BOARD/slr${idx}/dsp_loc_wangxi.tcl
        source $SDA_PATH/scripts/constraints/physical/$BOARD/slr${idx}/mss.tcl
        source $SDA_PATH/scripts/constraints/physical/$BOARD/slr${idx}/other_locs.tcl
} else {
    for { set idx 0}  {$idx < $DPU_NUM} {incr idx} { 
        set DPU_V3_TOP [expr \$SLR${idx}_DPU_V3_TOP]
        puts "Start to source timing constraints for [expr \$SLR${idx}_DPU_V3_TOP]"
        source $SDA_PATH/scripts/constraints/physical/$BOARD/slr${idx}/bram_loc_wangxi.tcl
        source $SDA_PATH/scripts/constraints/physical/$BOARD/slr${idx}/uram_loc_wangxi.tcl
        source $SDA_PATH/scripts/constraints/physical/$BOARD/slr${idx}/dsp_loc_wangxi.tcl
        source $SDA_PATH/scripts/constraints/physical/$BOARD/slr${idx}/mss.tcl
        if { $BOARD == "u200" } {
            source $SDA_PATH/scripts/constraints/physical/$BOARD/slr${idx}/other_locs.tcl  
        }
     }
}
}

if { ($SHELL_VER=="202002")||($SHELL_VER=="microsoft") } {
if { $DPU_NUM==1 } {
    set DPU_V3_TOP [expr \$SLR${SLR}_DPU_V3_TOP]
    puts "Start to source timing constraints for [expr \$SLR${SLR}_DPU_V3_TOP]"
    source $SDA_PATH/scripts/constraints/physical/${BOARD}_shell202/slr${SLR}/bram_loc_wangxi.tcl
    source $SDA_PATH/scripts/constraints/physical/${BOARD}_shell202/slr${SLR}/uram_loc_wangxi.tcl
    source $SDA_PATH/scripts/constraints/physical/${BOARD}_shell202/slr${SLR}/dsp_loc_wangxi.tcl
    source $SDA_PATH/scripts/constraints/physical/${BOARD}_shell202/slr${SLR}/mss.tcl
#    if { $BOARD == "u200" } {
#        source $SDA_PATH/scripts/constraints/physical/${BOARD}_shell202/slr${SLR}/other_locs.tcl  
#    }    
} elseif { ($DPU_NUM==2 )} {
        set idx 0
        set DPU_V3_TOP [expr \$SLR${idx}_DPU_V3_TOP]
        puts "Start to source timing constraints for [expr \$SLR${idx}_DPU_V3_TOP]"
        source $SDA_PATH/scripts/constraints/physical/${BOARD}_shell202/slr${idx}/bram_loc_wangxi.tcl
        source $SDA_PATH/scripts/constraints/physical/${BOARD}_shell202/slr${idx}/uram_loc_wangxi.tcl
        source $SDA_PATH/scripts/constraints/physical/${BOARD}_shell202/slr${idx}/dsp_loc_wangxi.tcl
        source $SDA_PATH/scripts/constraints/physical/${BOARD}_shell202/slr${idx}/mss.tcl
#        source $SDA_PATH/scripts/constraints/physical/${BOARD}_shell202/slr${idx}/other_locs.tcl
        set idx 2
        set DPU_V3_TOP [expr \$SLR${idx}_DPU_V3_TOP]
        puts "Start to source timing constraints for [expr \$SLR${idx}_DPU_V3_TOP]"
        source $SDA_PATH/scripts/constraints/physical/${BOARD}_shell202/slr${idx}/bram_loc_wangxi.tcl
        source $SDA_PATH/scripts/constraints/physical/${BOARD}_shell202/slr${idx}/uram_loc_wangxi.tcl
        source $SDA_PATH/scripts/constraints/physical/${BOARD}_shell202/slr${idx}/dsp_loc_wangxi.tcl
        source $SDA_PATH/scripts/constraints/physical/${BOARD}_shell202/slr${idx}/mss.tcl
#        source $SDA_PATH/scripts/constraints/physical/${BOARD}_shell202/slr${idx}/other_locs.tcl
} elseif {($DPU_NUM==3 )} {
        set idx 0
        set DPU_V3_TOP [expr \$SLR${idx}_DPU_V3_TOP]
        puts "Start to source timing constraints for [expr \$SLR${idx}_DPU_V3_TOP]"
        source $SDA_PATH/scripts/constraints/physical/${BOARD}_shell202/slr${idx}/bram_loc_wangxi.tcl
        source $SDA_PATH/scripts/constraints/physical/${BOARD}_shell202/slr${idx}/uram_loc_wangxi.tcl
        source $SDA_PATH/scripts/constraints/physical/${BOARD}_shell202/slr${idx}/dsp_loc_wangxi.tcl
        source $SDA_PATH/scripts/constraints/physical/${BOARD}_shell202/slr${idx}/mss.tcl
#        source $SDA_PATH/scripts/constraints/physical/${BOARD}_shell202/slr${idx}/other_locs.tcl
        set idx 2
        set DPU_V3_TOP [expr \$SLR${idx}_DPU_V3_TOP]
        puts "Start to source timing constraints for [expr \$SLR${idx}_DPU_V3_TOP]"
        source $SDA_PATH/scripts/constraints/physical/${BOARD}_shell202/slr${idx}/bram_loc_wangxi.tcl
        source $SDA_PATH/scripts/constraints/physical/${BOARD}_shell202/slr${idx}/uram_loc_wangxi.tcl
        source $SDA_PATH/scripts/constraints/physical/${BOARD}_shell202/slr${idx}/dsp_loc_wangxi.tcl
        source $SDA_PATH/scripts/constraints/physical/${BOARD}_shell202/slr${idx}/mss.tcl
#        source $SDA_PATH/scripts/constraints/physical/${BOARD}_shell202/slr${idx}/other_locs.tcl
        set idx 3
        set DPU_V3_TOP [expr \$SLR${idx}_DPU_V3_TOP]
        puts "Start to source timing constraints for [expr \$SLR${idx}_DPU_V3_TOP]"
        source $SDA_PATH/scripts/constraints/physical/${BOARD}_shell202/slr${idx}/bram_loc_wangxi.tcl
        source $SDA_PATH/scripts/constraints/physical/${BOARD}_shell202/slr${idx}/uram_loc_wangxi.tcl
        source $SDA_PATH/scripts/constraints/physical/${BOARD}_shell202/slr${idx}/dsp_loc_wangxi.tcl
        source $SDA_PATH/scripts/constraints/physical/${BOARD}_shell202/slr${idx}/mss.tcl
#        source $SDA_PATH/scripts/constraints/physical/${BOARD}_shell202/slr${idx}/other_locs.tcl
}
}



if { $BOARD == "u250" } {
    set_clock_uncertainty -setup 0.1 [get_clocks SLR*_ACLK_DR]
    set_clock_uncertainty -setup 0.2 [get_clocks SLR*_ACLK]
}

#if { ($SHELL_VER =="202002")&&($BOARD == "u200") } {
#    set_clock_uncertainty -setup 0.1 [get_clocks SLR*_ACLK_DR]
#    set_clock_uncertainty -setup 0.2 [get_clocks SLR*_ACLK]
#}

#####################################
#  Original opt_pre.tcl settings

make_outputs "opt_design"
