

proc gen_ila { ILA_INSTNAME PROB_NUM PROB_WIDTH } {
    global path_to_tmp_project

    set i 0
    set  WIDTH_CFG [list CONFIG.C_NUM_OF_PROBES $PROB_NUM]
    foreach WIDTH $PROB_WIDTH {
        lappend  WIDTH_CFG CONFIG.C_PROBE[expr $i]_WIDTH $WIDTH
        set i [expr $i+1]
    }

    create_ip -name ila -vendor xilinx.com -library ip -module_name $ILA_INSTNAME
    set_property -dict $WIDTH_CFG [get_ips $ILA_INSTNAME]
    generate_target {instantiation_template} [get_files $path_to_tmp_project/kernel_pack.srcs/sources_1/ip/$ILA_INSTNAME/$ILA_INSTNAME.xci]
    generate_target all [get_files  $path_to_tmp_project/kernel_pack.srcs/sources_1/ip/$ILA_INSTNAME/$ILA_INSTNAME.xci]

    read_ip [glob \
    $path_to_tmp_project/kernel_pack.srcs/sources_1/ip/$ILA_INSTNAME/$ILA_INSTNAME.xci \
    ]
}
