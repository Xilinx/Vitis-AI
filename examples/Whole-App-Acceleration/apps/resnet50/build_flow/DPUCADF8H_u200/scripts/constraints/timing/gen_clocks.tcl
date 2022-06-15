###########################################
#  clock region for DPUV3 in SLR0

puts "Start to source [info script]"


if { $DPU_NUM==1 } {
    set TOP \$SLR${SLR}_DPU_V3_WRAP_TOP
    if { $SHELL_VER=="201803" } {
        create_generated_clock -name SLR${SLR}_ACLK    [get_pins [expr $TOP]/BUFGCE_DIV_2/O] 
        create_generated_clock -name SLR${SLR}_ACLK_DR [get_pins [expr $TOP]/u_clk_wiz/inst/mmcme4_adv_inst/CLKOUT0]
        set_property CLOCK_DELAY_GROUP GROUP_SLR${SLR} [get_nets [expr $TOP]/u_clk_wiz/inst/clk_out_clk_wiz] [get_nets [expr $TOP]/dpdpuv3_top_data_aclk]
        set_property CLOCK_DEDICATED_ROUTE FALSE       [get_nets [expr $TOP]/u_clk_wiz/inst/clk_out_clk_wiz]
    } elseif { ($SHELL_VER=="202002")||($SHELL_VER=="microsoft") } {
        create_generated_clock -name SLR${SLR}_ACLK    [get_pins [expr $TOP]/BUFGCE_DIV_2/O] 
        create_generated_clock -name SLR${SLR}_ACLK_DR [get_pins [expr $TOP]/u_clk_wiz/inst/mmcme4_adv_inst/CLKOUT0]
        set_property CLOCK_DELAY_GROUP GROUP_SLR${SLR} [get_nets [expr $TOP]/u_clk_wiz/inst/clk_out_clk_wiz] [get_nets [expr $TOP]/dpdpuv3_top_data_aclk]
        set_property CLOCK_DEDICATED_ROUTE FALSE       [get_nets [expr $TOP]/u_clk_wiz/inst/clk_out_clk_wiz]
    } elseif { $SHELL_VER=="aws" } {
        create_generated_clock -name SLR${SLR}_ACLK    [get_pins [expr $TOP]/BUFGCE_DIV_2/O] 
        create_generated_clock -name SLR${SLR}_ACLK_DR [get_pins [expr $TOP]/u_clk_wiz/inst/mmcme4_adv_inst/CLKOUT0]
        set_property CLOCK_DELAY_GROUP GROUP_SLR${SLR} [get_nets [expr $TOP]/u_clk_wiz/inst/clk_out_clk_wiz] [get_nets [expr $TOP]/dpdpuv3_top_data_aclk]
        set_property CLOCK_DEDICATED_ROUTE FALSE       [get_nets [expr $TOP]/u_clk_wiz/inst/clk_out_clk_wiz]
    }
} elseif { ($DPU_NUM==2 )&&($BOARD=="u200")} {
        set idx 0
        set TOP \$SLR${idx}_DPU_V3_WRAP_TOP
        create_generated_clock -name SLR${idx}_ACLK    [get_pins [expr $TOP]/BUFGCE_DIV_2/O] 
        create_generated_clock -name SLR${idx}_ACLK_DR [get_pins [expr $TOP]/u_clk_wiz/inst/mmcme4_adv_inst/CLKOUT0]
        set_property CLOCK_DELAY_GROUP GROUP_SLR${idx} [get_nets [expr $TOP]/u_clk_wiz/inst/clk_out_clk_wiz] [get_nets [expr $TOP]/dpdpuv3_top_data_aclk]
        set_property CLOCK_DEDICATED_ROUTE FALSE       [get_nets [expr $TOP]/u_clk_wiz/inst/clk_out_clk_wiz]
        set idx 2
        set TOP \$SLR${idx}_DPU_V3_WRAP_TOP
        create_generated_clock -name SLR${idx}_ACLK    [get_pins [expr $TOP]/BUFGCE_DIV_2/O] 
        create_generated_clock -name SLR${idx}_ACLK_DR [get_pins [expr $TOP]/u_clk_wiz/inst/mmcme4_adv_inst/CLKOUT0]
        set_property CLOCK_DELAY_GROUP GROUP_SLR${idx} [get_nets [expr $TOP]/u_clk_wiz/inst/clk_out_clk_wiz] [get_nets [expr $TOP]/dpdpuv3_top_data_aclk]
        set_property CLOCK_DEDICATED_ROUTE FALSE       [get_nets [expr $TOP]/u_clk_wiz/inst/clk_out_clk_wiz]
} elseif {($BOARD=="u250")&&(($SHELL_VER=="202002")||($SHELL_VER=="microsoft"))&&($DPU_NUM==2 )} {
        set idx 0
        set TOP \$SLR${idx}_DPU_V3_WRAP_TOP
        create_generated_clock -name SLR${idx}_ACLK    [get_pins [expr $TOP]/BUFGCE_DIV_2/O] 
        create_generated_clock -name SLR${idx}_ACLK_DR [get_pins [expr $TOP]/u_clk_wiz/inst/mmcme4_adv_inst/CLKOUT0]
        set_property CLOCK_DELAY_GROUP GROUP_SLR${idx} [get_nets [expr $TOP]/u_clk_wiz/inst/clk_out_clk_wiz] [get_nets [expr $TOP]/dpdpuv3_top_data_aclk]
        set_property CLOCK_DEDICATED_ROUTE FALSE       [get_nets [expr $TOP]/u_clk_wiz/inst/clk_out_clk_wiz]
        set idx 2
        set TOP \$SLR${idx}_DPU_V3_WRAP_TOP
        create_generated_clock -name SLR${idx}_ACLK    [get_pins [expr $TOP]/BUFGCE_DIV_2/O] 
        create_generated_clock -name SLR${idx}_ACLK_DR [get_pins [expr $TOP]/u_clk_wiz/inst/mmcme4_adv_inst/CLKOUT0]
        set_property CLOCK_DELAY_GROUP GROUP_SLR${idx} [get_nets [expr $TOP]/u_clk_wiz/inst/clk_out_clk_wiz] [get_nets [expr $TOP]/dpdpuv3_top_data_aclk]
        set_property CLOCK_DEDICATED_ROUTE FALSE       [get_nets [expr $TOP]/u_clk_wiz/inst/clk_out_clk_wiz]
} elseif {($BOARD=="u250")&&(($SHELL_VER=="202002")||($SHELL_VER=="microsoft"))&&($DPU_NUM==3 )} {
        set idx 0
        set TOP \$SLR${idx}_DPU_V3_WRAP_TOP
        create_generated_clock -name SLR${idx}_ACLK    [get_pins [expr $TOP]/BUFGCE_DIV_2/O] 
        create_generated_clock -name SLR${idx}_ACLK_DR [get_pins [expr $TOP]/u_clk_wiz/inst/mmcme4_adv_inst/CLKOUT0]
        set_property CLOCK_DELAY_GROUP GROUP_SLR${idx} [get_nets [expr $TOP]/u_clk_wiz/inst/clk_out_clk_wiz] [get_nets [expr $TOP]/dpdpuv3_top_data_aclk]
        set_property CLOCK_DEDICATED_ROUTE FALSE       [get_nets [expr $TOP]/u_clk_wiz/inst/clk_out_clk_wiz]
        set idx 2
        set TOP \$SLR${idx}_DPU_V3_WRAP_TOP
        create_generated_clock -name SLR${idx}_ACLK    [get_pins [expr $TOP]/BUFGCE_DIV_2/O] 
        create_generated_clock -name SLR${idx}_ACLK_DR [get_pins [expr $TOP]/u_clk_wiz/inst/mmcme4_adv_inst/CLKOUT0]
        set_property CLOCK_DELAY_GROUP GROUP_SLR${idx} [get_nets [expr $TOP]/u_clk_wiz/inst/clk_out_clk_wiz] [get_nets [expr $TOP]/dpdpuv3_top_data_aclk]
        set_property CLOCK_DEDICATED_ROUTE FALSE       [get_nets [expr $TOP]/u_clk_wiz/inst/clk_out_clk_wiz]
        set idx 3
        set TOP \$SLR${idx}_DPU_V3_WRAP_TOP
        create_generated_clock -name SLR${idx}_ACLK    [get_pins [expr $TOP]/BUFGCE_DIV_2/O] 
        create_generated_clock -name SLR${idx}_ACLK_DR [get_pins [expr $TOP]/u_clk_wiz/inst/mmcme4_adv_inst/CLKOUT0]
        set_property CLOCK_DELAY_GROUP GROUP_SLR${idx} [get_nets [expr $TOP]/u_clk_wiz/inst/clk_out_clk_wiz] [get_nets [expr $TOP]/dpdpuv3_top_data_aclk]
        set_property CLOCK_DEDICATED_ROUTE FALSE       [get_nets [expr $TOP]/u_clk_wiz/inst/clk_out_clk_wiz]
} else {
    for { set idx 0}  {$idx < $DPU_NUM} {incr idx} { 
        set TOP \$SLR${idx}_DPU_V3_WRAP_TOP
        create_generated_clock -name SLR${idx}_ACLK    [get_pins [expr $TOP]/BUFGCE_DIV_2/O] 
        create_generated_clock -name SLR${idx}_ACLK_DR [get_pins [expr $TOP]/u_clk_wiz/inst/mmcme4_adv_inst/CLKOUT0]
        set_property CLOCK_DELAY_GROUP GROUP_SLR${idx} [get_nets [expr $TOP]/u_clk_wiz/inst/clk_out_clk_wiz] [get_nets [expr $TOP]/dpdpuv3_top_data_aclk]
        set_property CLOCK_DEDICATED_ROUTE FALSE       [get_nets [expr $TOP]/u_clk_wiz/inst/clk_out_clk_wiz]
    }
}

