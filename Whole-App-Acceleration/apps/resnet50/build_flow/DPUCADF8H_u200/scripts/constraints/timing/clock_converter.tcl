###############################################################################################################
# Core-Level Timing Constraints for axi_clock_converter Component "axi_clock_converter_0"
###############################################################################################################
#
# This component is configured to perform asynchronous clock-domain-crossing.
# In order for these core-level constraints to work properly, 
# the following rules apply to your system-level timing constraints:
#   1. Each of the nets connected to the s_axi_aclk and m_axi_aclk ports of this component
#      must have exactly one clock defined on it, using either
#      a) a create_clock command on a top-level clock pin specified in your system XDC file, or
#      b) a create_generated_clock command, typically generated automatically by a core 
#          producing a derived clock signal.
#   2. The s_axi_aclk and m_axi_aclk ports of this component should not be connected to the
#      same clock source.
#

puts "Start to source [info script]"


if { $DPU_NUM==1 } {
    set s_ram_cells [filter [all_fanout -from [get_nets [expr \$SLR${SLR}_DPU_V3_WRAP_TOP]/u_axi_clock_converter_512/s_axi_aclk] -flat -endpoints_only -only_cells] {PRIMITIVE_SUBGROUP==dram || PRIMITIVE_SUBGROUP==LUTRAM}]
    set m_ram_cells [filter [all_fanout -from [get_nets [expr \$SLR${SLR}_DPU_V3_WRAP_TOP]/u_axi_clock_converter_512/m_axi_aclk] -flat -endpoints_only -only_cells] {PRIMITIVE_SUBGROUP==dram || PRIMITIVE_SUBGROUP==LUTRAM}]
    set_false_path -from [get_pins -of $s_ram_cells -filter {REF_PIN_NAME == CLK}] -through [get_pins -of $s_ram_cells -filter {REF_PIN_NAME == O}] 
    set_false_path -from [get_pins -of $m_ram_cells -filter {REF_PIN_NAME == CLK}] -through [get_pins -of $m_ram_cells -filter {REF_PIN_NAME == O}] 
} elseif { ($DPU_NUM==2 )&&($BOARD=="u200")} {
    set idx 0
    set TOP \$SLR${idx}_DPU_V3_WRAP_TOP
    set s_ram_cells [filter [all_fanout -from [get_nets [expr $TOP]/u_axi_clock_converter_512/s_axi_aclk] -flat -endpoints_only -only_cells] {PRIMITIVE_SUBGROUP==dram || PRIMITIVE_SUBGROUP==LUTRAM}]
    set m_ram_cells [filter [all_fanout -from [get_nets [expr $TOP]/u_axi_clock_converter_512/m_axi_aclk] -flat -endpoints_only -only_cells] {PRIMITIVE_SUBGROUP==dram || PRIMITIVE_SUBGROUP==LUTRAM}]
    set_false_path -from [get_pins -of $s_ram_cells -filter {REF_PIN_NAME == CLK}] -through [get_pins -of $s_ram_cells -filter {REF_PIN_NAME == O}] 
    set_false_path -from [get_pins -of $m_ram_cells -filter {REF_PIN_NAME == CLK}] -through [get_pins -of $m_ram_cells -filter {REF_PIN_NAME == O}] 
    set idx 2
    set TOP \$SLR${idx}_DPU_V3_WRAP_TOP
    set s_ram_cells [filter [all_fanout -from [get_nets [expr $TOP]/u_axi_clock_converter_512/s_axi_aclk] -flat -endpoints_only -only_cells] {PRIMITIVE_SUBGROUP==dram || PRIMITIVE_SUBGROUP==LUTRAM}]
    set m_ram_cells [filter [all_fanout -from [get_nets [expr $TOP]/u_axi_clock_converter_512/m_axi_aclk] -flat -endpoints_only -only_cells] {PRIMITIVE_SUBGROUP==dram || PRIMITIVE_SUBGROUP==LUTRAM}]
    set_false_path -from [get_pins -of $s_ram_cells -filter {REF_PIN_NAME == CLK}] -through [get_pins -of $s_ram_cells -filter {REF_PIN_NAME == O}] 
    set_false_path -from [get_pins -of $m_ram_cells -filter {REF_PIN_NAME == CLK}] -through [get_pins -of $m_ram_cells -filter {REF_PIN_NAME == O}] 
} elseif {($BOARD=="u250")&&(($SHELL_VER=="202002")||($SHELL_VER=="microsoft"))&&($DPU_NUM==2 )} {
    set idx 0
    set TOP \$SLR${idx}_DPU_V3_WRAP_TOP
    set s_ram_cells [filter [all_fanout -from [get_nets [expr $TOP]/u_axi_clock_converter_512/s_axi_aclk] -flat -endpoints_only -only_cells] {PRIMITIVE_SUBGROUP==dram || PRIMITIVE_SUBGROUP==LUTRAM}]
    set m_ram_cells [filter [all_fanout -from [get_nets [expr $TOP]/u_axi_clock_converter_512/m_axi_aclk] -flat -endpoints_only -only_cells] {PRIMITIVE_SUBGROUP==dram || PRIMITIVE_SUBGROUP==LUTRAM}]
    set_false_path -from [get_pins -of $s_ram_cells -filter {REF_PIN_NAME == CLK}] -through [get_pins -of $s_ram_cells -filter {REF_PIN_NAME == O}] 
    set_false_path -from [get_pins -of $m_ram_cells -filter {REF_PIN_NAME == CLK}] -through [get_pins -of $m_ram_cells -filter {REF_PIN_NAME == O}] 
    set idx 2
    set TOP \$SLR${idx}_DPU_V3_WRAP_TOP
    set s_ram_cells [filter [all_fanout -from [get_nets [expr $TOP]/u_axi_clock_converter_512/s_axi_aclk] -flat -endpoints_only -only_cells] {PRIMITIVE_SUBGROUP==dram || PRIMITIVE_SUBGROUP==LUTRAM}]
    set m_ram_cells [filter [all_fanout -from [get_nets [expr $TOP]/u_axi_clock_converter_512/m_axi_aclk] -flat -endpoints_only -only_cells] {PRIMITIVE_SUBGROUP==dram || PRIMITIVE_SUBGROUP==LUTRAM}]
    set_false_path -from [get_pins -of $s_ram_cells -filter {REF_PIN_NAME == CLK}] -through [get_pins -of $s_ram_cells -filter {REF_PIN_NAME == O}] 
    set_false_path -from [get_pins -of $m_ram_cells -filter {REF_PIN_NAME == CLK}] -through [get_pins -of $m_ram_cells -filter {REF_PIN_NAME == O}] 
} elseif {($BOARD=="u250")&&(($SHELL_VER=="202002")||($SHELL_VER=="microsoft"))&&($DPU_NUM==3 )} {
    set idx 0
    set TOP \$SLR${idx}_DPU_V3_WRAP_TOP
    set s_ram_cells [filter [all_fanout -from [get_nets [expr $TOP]/u_axi_clock_converter_512/s_axi_aclk] -flat -endpoints_only -only_cells] {PRIMITIVE_SUBGROUP==dram || PRIMITIVE_SUBGROUP==LUTRAM}]
    set m_ram_cells [filter [all_fanout -from [get_nets [expr $TOP]/u_axi_clock_converter_512/m_axi_aclk] -flat -endpoints_only -only_cells] {PRIMITIVE_SUBGROUP==dram || PRIMITIVE_SUBGROUP==LUTRAM}]
    set_false_path -from [get_pins -of $s_ram_cells -filter {REF_PIN_NAME == CLK}] -through [get_pins -of $s_ram_cells -filter {REF_PIN_NAME == O}] 
    set_false_path -from [get_pins -of $m_ram_cells -filter {REF_PIN_NAME == CLK}] -through [get_pins -of $m_ram_cells -filter {REF_PIN_NAME == O}] 
    set idx 2
    set TOP \$SLR${idx}_DPU_V3_WRAP_TOP
    set s_ram_cells [filter [all_fanout -from [get_nets [expr $TOP]/u_axi_clock_converter_512/s_axi_aclk] -flat -endpoints_only -only_cells] {PRIMITIVE_SUBGROUP==dram || PRIMITIVE_SUBGROUP==LUTRAM}]
    set m_ram_cells [filter [all_fanout -from [get_nets [expr $TOP]/u_axi_clock_converter_512/m_axi_aclk] -flat -endpoints_only -only_cells] {PRIMITIVE_SUBGROUP==dram || PRIMITIVE_SUBGROUP==LUTRAM}]
    set_false_path -from [get_pins -of $s_ram_cells -filter {REF_PIN_NAME == CLK}] -through [get_pins -of $s_ram_cells -filter {REF_PIN_NAME == O}] 
    set_false_path -from [get_pins -of $m_ram_cells -filter {REF_PIN_NAME == CLK}] -through [get_pins -of $m_ram_cells -filter {REF_PIN_NAME == O}] 
    set idx 3
    set TOP \$SLR${idx}_DPU_V3_WRAP_TOP
    set s_ram_cells [filter [all_fanout -from [get_nets [expr $TOP]/u_axi_clock_converter_512/s_axi_aclk] -flat -endpoints_only -only_cells] {PRIMITIVE_SUBGROUP==dram || PRIMITIVE_SUBGROUP==LUTRAM}]
    set m_ram_cells [filter [all_fanout -from [get_nets [expr $TOP]/u_axi_clock_converter_512/m_axi_aclk] -flat -endpoints_only -only_cells] {PRIMITIVE_SUBGROUP==dram || PRIMITIVE_SUBGROUP==LUTRAM}]
    set_false_path -from [get_pins -of $s_ram_cells -filter {REF_PIN_NAME == CLK}] -through [get_pins -of $s_ram_cells -filter {REF_PIN_NAME == O}] 
    set_false_path -from [get_pins -of $m_ram_cells -filter {REF_PIN_NAME == CLK}] -through [get_pins -of $m_ram_cells -filter {REF_PIN_NAME == O}] 
} else {
    for { set idx 0}  {$idx < $DPU_NUM} {incr idx} { 
        if { ($SHELL_VER!="202002")||($idx!=1 ) } {
        set TOP \$SLR${idx}_DPU_V3_WRAP_TOP
        set s_ram_cells [filter [all_fanout -from [get_nets [expr $TOP]/u_axi_clock_converter_512/s_axi_aclk] -flat -endpoints_only -only_cells] {PRIMITIVE_SUBGROUP==dram || PRIMITIVE_SUBGROUP==LUTRAM}]
        set m_ram_cells [filter [all_fanout -from [get_nets [expr $TOP]/u_axi_clock_converter_512/m_axi_aclk] -flat -endpoints_only -only_cells] {PRIMITIVE_SUBGROUP==dram || PRIMITIVE_SUBGROUP==LUTRAM}]
        set_false_path -from [get_pins -of $s_ram_cells -filter {REF_PIN_NAME == CLK}] -through [get_pins -of $s_ram_cells -filter {REF_PIN_NAME == O}] 
        set_false_path -from [get_pins -of $m_ram_cells -filter {REF_PIN_NAME == CLK}] -through [get_pins -of $m_ram_cells -filter {REF_PIN_NAME == O}] 
        } 
    }
}


