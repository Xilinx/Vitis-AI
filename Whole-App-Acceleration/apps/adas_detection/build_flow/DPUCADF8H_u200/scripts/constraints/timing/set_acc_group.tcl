puts "Start to source [info script]"



if { $DPU_NUM==1 } {
    set TOP \$SLR${SLR}_DPU_V3_WRAP_TOP
    if { $SHELL_VER=="201803" } {
        group_path -from [get_pins -of [get_cells -hier -filter {PRIMITIVE_TYPE =~ REGISTER.SDR.*}] -filter name=~[expr $TOP]/u_dpdpuv3_top/core_array[*].u_core/u_conv_top_core/u_conv_calc_part/gen_dsp_block_col[*].u_conv_dsp_block_i/gen_dsp_chain_col[*].u_conv_dsp_chain_i/u_conv_dsp_wide_acc/inst_acc_hi/*/C] -name group_acc_hi
        group_path -from [get_pins -of [get_cells -hier -filter {PRIMITIVE_TYPE =~ REGISTER.SDR.*}] -filter name=~[expr $TOP]/u_dpdpuv3_top/core_array[*].u_core/u_conv_top_core/u_conv_calc_part/gen_dsp_block_col[*].u_conv_dsp_block_i/gen_dsp_chain_col[*].u_conv_dsp_chain_i/u_conv_dsp_wide_acc/inst_acc_lo/*/C] -name group_acc_lo
    } elseif { ($SHELL_VER=="202002")||($SHELL_VER=="microsoft") } {
        group_path -from [get_pins -of [get_cells -hier -filter {PRIMITIVE_TYPE =~ REGISTER.SDR.*}] -filter name=~[expr $TOP]/u_dpdpuv3_top/core_array[*].u_core/u_conv_top_core/u_conv_calc_part/gen_dsp_block_col[*].u_conv_dsp_block_i/gen_dsp_chain_col[*].u_conv_dsp_chain_i/u_conv_dsp_wide_acc/inst_acc_hi/*/C] -name group_acc_hi
        group_path -from [get_pins -of [get_cells -hier -filter {PRIMITIVE_TYPE =~ REGISTER.SDR.*}] -filter name=~[expr $TOP]/u_dpdpuv3_top/core_array[*].u_core/u_conv_top_core/u_conv_calc_part/gen_dsp_block_col[*].u_conv_dsp_block_i/gen_dsp_chain_col[*].u_conv_dsp_chain_i/u_conv_dsp_wide_acc/inst_acc_lo/*/C] -name group_acc_lo
    } elseif { ($SHELL_VER=="aws") } {
        group_path -from [get_pins -of [get_cells -hier -filter {PRIMITIVE_TYPE =~ REGISTER.SDR.*}] -filter name=~[expr $TOP]/u_dpdpuv3_top/core_array[*].u_core/u_conv_top_core/u_conv_calc_part/gen_dsp_block_col[*].u_conv_dsp_block_i/gen_dsp_chain_col[*].u_conv_dsp_chain_i/u_conv_dsp_wide_acc/inst_acc_hi/*/C] -name group_acc_hi
        group_path -from [get_pins -of [get_cells -hier -filter {PRIMITIVE_TYPE =~ REGISTER.SDR.*}] -filter name=~[expr $TOP]/u_dpdpuv3_top/core_array[*].u_core/u_conv_top_core/u_conv_calc_part/gen_dsp_block_col[*].u_conv_dsp_block_i/gen_dsp_chain_col[*].u_conv_dsp_chain_i/u_conv_dsp_wide_acc/inst_acc_lo/*/C] -name group_acc_lo
    }
} elseif { ($DPU_NUM==2 )&&($BOARD=="u200")} {
        set idx 0
        set TOP \$SLR${idx}_DPU_V3_WRAP_TOP
        group_path -from [get_pins -of [get_cells -hier -filter {PRIMITIVE_TYPE =~ REGISTER.SDR.*}] -filter name=~[expr $TOP]/u_dpdpuv3_top/core_array[*].u_core/u_conv_top_core/u_conv_calc_part/gen_dsp_block_col[*].u_conv_dsp_block_i/gen_dsp_chain_col[*].u_conv_dsp_chain_i/u_conv_dsp_wide_acc/inst_acc_hi/*/C] -name group_acc_hi
        group_path -from [get_pins -of [get_cells -hier -filter {PRIMITIVE_TYPE =~ REGISTER.SDR.*}] -filter name=~[expr $TOP]/u_dpdpuv3_top/core_array[*].u_core/u_conv_top_core/u_conv_calc_part/gen_dsp_block_col[*].u_conv_dsp_block_i/gen_dsp_chain_col[*].u_conv_dsp_chain_i/u_conv_dsp_wide_acc/inst_acc_lo/*/C] -name group_acc_lo
        set idx 2
        set TOP \$SLR${idx}_DPU_V3_WRAP_TOP
        group_path -from [get_pins -of [get_cells -hier -filter {PRIMITIVE_TYPE =~ REGISTER.SDR.*}] -filter name=~[expr $TOP]/u_dpdpuv3_top/core_array[*].u_core/u_conv_top_core/u_conv_calc_part/gen_dsp_block_col[*].u_conv_dsp_block_i/gen_dsp_chain_col[*].u_conv_dsp_chain_i/u_conv_dsp_wide_acc/inst_acc_hi/*/C] -name group_acc_hi
        group_path -from [get_pins -of [get_cells -hier -filter {PRIMITIVE_TYPE =~ REGISTER.SDR.*}] -filter name=~[expr $TOP]/u_dpdpuv3_top/core_array[*].u_core/u_conv_top_core/u_conv_calc_part/gen_dsp_block_col[*].u_conv_dsp_block_i/gen_dsp_chain_col[*].u_conv_dsp_chain_i/u_conv_dsp_wide_acc/inst_acc_lo/*/C] -name group_acc_lo
} elseif {($BOARD=="u250")&&(($SHELL_VER=="202002")||($SHELL_VER=="microsoft"))&&($DPU_NUM==2 )} {
        set idx 0
        set TOP \$SLR${idx}_DPU_V3_WRAP_TOP
        group_path -from [get_pins -of [get_cells -hier -filter {PRIMITIVE_TYPE =~ REGISTER.SDR.*}] -filter name=~[expr $TOP]/u_dpdpuv3_top/core_array[*].u_core/u_conv_top_core/u_conv_calc_part/gen_dsp_block_col[*].u_conv_dsp_block_i/gen_dsp_chain_col[*].u_conv_dsp_chain_i/u_conv_dsp_wide_acc/inst_acc_hi/*/C] -name group_acc_hi
        group_path -from [get_pins -of [get_cells -hier -filter {PRIMITIVE_TYPE =~ REGISTER.SDR.*}] -filter name=~[expr $TOP]/u_dpdpuv3_top/core_array[*].u_core/u_conv_top_core/u_conv_calc_part/gen_dsp_block_col[*].u_conv_dsp_block_i/gen_dsp_chain_col[*].u_conv_dsp_chain_i/u_conv_dsp_wide_acc/inst_acc_lo/*/C] -name group_acc_lo
        set idx 2
        set TOP \$SLR${idx}_DPU_V3_WRAP_TOP
        group_path -from [get_pins -of [get_cells -hier -filter {PRIMITIVE_TYPE =~ REGISTER.SDR.*}] -filter name=~[expr $TOP]/u_dpdpuv3_top/core_array[*].u_core/u_conv_top_core/u_conv_calc_part/gen_dsp_block_col[*].u_conv_dsp_block_i/gen_dsp_chain_col[*].u_conv_dsp_chain_i/u_conv_dsp_wide_acc/inst_acc_hi/*/C] -name group_acc_hi
        group_path -from [get_pins -of [get_cells -hier -filter {PRIMITIVE_TYPE =~ REGISTER.SDR.*}] -filter name=~[expr $TOP]/u_dpdpuv3_top/core_array[*].u_core/u_conv_top_core/u_conv_calc_part/gen_dsp_block_col[*].u_conv_dsp_block_i/gen_dsp_chain_col[*].u_conv_dsp_chain_i/u_conv_dsp_wide_acc/inst_acc_lo/*/C] -name group_acc_lo
} elseif {($BOARD=="u250")&&(($SHELL_VER=="202002")||($SHELL_VER=="microsoft"))&&($DPU_NUM==3 )} {
        set idx 0
        set TOP \$SLR${idx}_DPU_V3_WRAP_TOP
        group_path -from [get_pins -of [get_cells -hier -filter {PRIMITIVE_TYPE =~ REGISTER.SDR.*}] -filter name=~[expr $TOP]/u_dpdpuv3_top/core_array[*].u_core/u_conv_top_core/u_conv_calc_part/gen_dsp_block_col[*].u_conv_dsp_block_i/gen_dsp_chain_col[*].u_conv_dsp_chain_i/u_conv_dsp_wide_acc/inst_acc_hi/*/C] -name group_acc_hi
        group_path -from [get_pins -of [get_cells -hier -filter {PRIMITIVE_TYPE =~ REGISTER.SDR.*}] -filter name=~[expr $TOP]/u_dpdpuv3_top/core_array[*].u_core/u_conv_top_core/u_conv_calc_part/gen_dsp_block_col[*].u_conv_dsp_block_i/gen_dsp_chain_col[*].u_conv_dsp_chain_i/u_conv_dsp_wide_acc/inst_acc_lo/*/C] -name group_acc_lo
        set idx 2
        set TOP \$SLR${idx}_DPU_V3_WRAP_TOP
        group_path -from [get_pins -of [get_cells -hier -filter {PRIMITIVE_TYPE =~ REGISTER.SDR.*}] -filter name=~[expr $TOP]/u_dpdpuv3_top/core_array[*].u_core/u_conv_top_core/u_conv_calc_part/gen_dsp_block_col[*].u_conv_dsp_block_i/gen_dsp_chain_col[*].u_conv_dsp_chain_i/u_conv_dsp_wide_acc/inst_acc_hi/*/C] -name group_acc_hi
        group_path -from [get_pins -of [get_cells -hier -filter {PRIMITIVE_TYPE =~ REGISTER.SDR.*}] -filter name=~[expr $TOP]/u_dpdpuv3_top/core_array[*].u_core/u_conv_top_core/u_conv_calc_part/gen_dsp_block_col[*].u_conv_dsp_block_i/gen_dsp_chain_col[*].u_conv_dsp_chain_i/u_conv_dsp_wide_acc/inst_acc_lo/*/C] -name group_acc_lo
        set idx 3
        set TOP \$SLR${idx}_DPU_V3_WRAP_TOP
        group_path -from [get_pins -of [get_cells -hier -filter {PRIMITIVE_TYPE =~ REGISTER.SDR.*}] -filter name=~[expr $TOP]/u_dpdpuv3_top/core_array[*].u_core/u_conv_top_core/u_conv_calc_part/gen_dsp_block_col[*].u_conv_dsp_block_i/gen_dsp_chain_col[*].u_conv_dsp_chain_i/u_conv_dsp_wide_acc/inst_acc_hi/*/C] -name group_acc_hi
        group_path -from [get_pins -of [get_cells -hier -filter {PRIMITIVE_TYPE =~ REGISTER.SDR.*}] -filter name=~[expr $TOP]/u_dpdpuv3_top/core_array[*].u_core/u_conv_top_core/u_conv_calc_part/gen_dsp_block_col[*].u_conv_dsp_block_i/gen_dsp_chain_col[*].u_conv_dsp_chain_i/u_conv_dsp_wide_acc/inst_acc_lo/*/C] -name group_acc_lo
} else {
    for { set idx 0}  {$idx < $DPU_NUM} {incr idx} { 
        if { ($SHELL_VER!="202002")||($idx!=1 ) } {
        set TOP \$SLR${idx}_DPU_V3_WRAP_TOP
        group_path -from [get_pins -of [get_cells -hier -filter {PRIMITIVE_TYPE =~ REGISTER.SDR.*}] -filter name=~[expr $TOP]/u_dpdpuv3_top/core_array[*].u_core/u_conv_top_core/u_conv_calc_part/gen_dsp_block_col[*].u_conv_dsp_block_i/gen_dsp_chain_col[*].u_conv_dsp_chain_i/u_conv_dsp_wide_acc/inst_acc_hi/*/C] -name group_acc_hi
        group_path -from [get_pins -of [get_cells -hier -filter {PRIMITIVE_TYPE =~ REGISTER.SDR.*}] -filter name=~[expr $TOP]/u_dpdpuv3_top/core_array[*].u_core/u_conv_top_core/u_conv_calc_part/gen_dsp_block_col[*].u_conv_dsp_block_i/gen_dsp_chain_col[*].u_conv_dsp_chain_i/u_conv_dsp_wide_acc/inst_acc_lo/*/C] -name group_acc_lo
        }
    }
}

