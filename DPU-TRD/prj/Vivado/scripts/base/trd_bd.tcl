# (c) Copyright 2018-2018 Xilinx, Inc. All rights reserved.
#
# This file contains confidential and proprietary information
# of Xilinx, Inc. and is protected under U.S. and
# international copyright and other intellectual property
# laws.
#
# DISCLAIMER
# This disclaimer is not a license and does not grant any
# rights to the materials distributed herewith. Except as
# otherwise provided in a valid license issued to you by
# Xilinx, and to the maximum extent permitted by applicable
# law: (1) THESE MATERIALS ARE MADE AVAILABLE "AS IS" AND
# WITH ALL FAULTS, AND XILINX HEREBY DISCLAIMS ALL WARRANTIES
# AND CONDITIONS, EXPRESS, IMPLIED, OR STATUTORY, INCLUDING
# BUT NOT LIMITED TO WARRANTIES OF MERCHANTABILITY, NON-
# INFRINGEMENT, OR FITNESS FOR ANY PARTICULAR PURPOSE; and
# (2) Xilinx shall not be liable (whether in contract or tort,
# including negligence, or under any other theory of
# liability) for any loss or damage of any kind or nature
# related to, arising under or in connection with these
# materials, including for any direct, or any indirect,
# special, incidental, or consequential loss or damage
# (including loss of data, profits, goodwill, or any type of
# loss or damage suffered as a result of any action brought
# by a third party) even if such damage or loss was
# reasonably foreseeable or Xilinx had been advised of the
# possibility of the same.
#
# CRITICAL APPLICATIONS
# Xilinx products are not designed or intended to be fail-
# safe, or for use in any application requiring fail-safe
# performance, such as life-support or safety devices or
# systems, Class III medical devices, nuclear facilities,
# applications related to the deployment of airbags, or any
# other applications that could lead to death, personal
# injury, or severe property or environmental damage
# (individually and collectively, "Critical
# Applications"). Customer assumes the sole risk and
# liability of any use of Xilinx products in Critical
# Applications, subject only to applicable laws and
# regulations governing limitations on product liability.
#
# THIS COPYRIGHT NOTICE AND DISCLAIMER MUST BE RETAINED AS
# PART OF THIS FILE AT ALL TIMES.
#

#****************************************************************
# set global dict_prj
#****************************************************************

#****************************************************************
# set environment
#****************************************************************
dict set dict_prj dict_sys pwd_dir            [pwd]
dict set dict_prj dict_sys srcs_dir           [file dirname [file dirname [file dirname [dict get $dict_prj dict_sys work_dir]]]]/dpu_ip
dict set dict_prj dict_sys ip_dir             [dict get $dict_prj dict_sys srcs_dir]/dpu_eu_v3_2_0
dict set dict_prj dict_sys xdc_dir            [file dirname [dict get $dict_prj dict_sys work_dir]]/constrs
dict set dict_prj dict_sys prj_dir            [file dirname [dict get $dict_prj dict_sys work_dir]]/prj
dict set dict_prj dict_sys bd_dir             [file dirname [dict get $dict_prj dict_sys work_dir]]/srcs
dict set dict_prj dict_sys bd_path            [dict get $dict_prj dict_sys bd_dir]/[dict get $dict_prj dict_sys bd_name]/[dict get $dict_prj dict_sys bd_name].bd
dict set dict_prj dict_sys bd_top_name        [dict get $dict_prj dict_sys bd_name]\_wrapper
dict set dict_prj dict_sys work_name          [file tail [dict get $dict_prj dict_sys  work_dir]]
dict set dict_prj dict_sys bd_wrapper_name    [dict get $dict_prj dict_sys bd_top_name]\.v
dict set dict_prj dict_sys bd_wrapper_path    [dict get $dict_prj dict_sys bd_dir]/[dict get $dict_prj dict_sys bd_name]/hdl/[dict get $dict_prj dict_sys bd_wrapper_name]
dict set dict_prj dict_sys prj_family         [string tolower [get_property FAMILY [get_parts [dict get $dict_prj dict_sys prj_part]]]]
dict set dict_prj dict_sys ver_vivado         [version -short]
dict set dict_prj dict_sys ver_year           [scan [clock format [clock seconds] -format "%Y"] %d]
dict set dict_prj dict_sys ver_month          [scan [clock format [clock seconds] -format "%m"] %d]
dict set dict_prj dict_sys ver_day            [scan [clock format [clock seconds] -format "%e"] %d]
dict set dict_prj dict_sys ver_hour           [scan [clock format [clock seconds] -format "%k"] %d]
dict set dict_prj dict_sys ver_bit            [expr [scan [clock format [clock seconds] -format "%M"] %d]/15]
dict set dict_prj dict_param  DPU_HP_CC_EN    [expr {([dict get $dict_prj dict_param DPU_CLK_MHz]<[dict get $dict_prj dict_param HP_CLK_MHz] )?{0}:{1} }]
dict set dict_prj dict_param  DPU_HP_DATA_BW  [expr {([dict get $dict_prj dict_sys prj_family]=={zynq})?{64}:{128}}]
dict set dict_prj dict_stgy synth synth_1     [dict create "STRATEGY" {}  "JOBS" 40 ]
dict set dict_prj dict_stgy impl impl_1_01 	  [dict create "PARENT" {} "STRATEGY" {}  "JOBS" 40 ]
dict set dict_prj dict_stgy impl impl_1_03 	  [dict create "PARENT" {} "STRATEGY" {}  "JOBS" 40 ]
dict set dict_prj dict_stgy impl impl_1_14 	  [dict create "PARENT" {} "STRATEGY" {}  "JOBS" 40 ]
dict set dict_prj dict_stgy impl impl_1_15 	  [dict create "PARENT" {} "STRATEGY" {}  "JOBS" 40 ]
dict set dict_prj dict_param  DPU_CONV_WP     {1}
#****************************************************************
# get dict value
#****************************************************************
proc lib_dict_value { args } {
  global dict_prj
  if { [dict exist $dict_prj {*}$args] != {0} } {
    return  [dict get $dict_prj {*}$args]
  } else {
    return {-1}
  }
}

#****************************************************************
# get dict_ip value
#****************************************************************
proc lib_value { module key } {
  return  [lib_dict_value dict_ip $module $key]
}

#****************************************************************
# get dict_ip path
#****************************************************************
proc lib_cell { module } {
  return  [lib_value $module PATH]
}

#****************************************************************
# get dict_ip path name
#****************************************************************
proc lib_cell_name { module } {
  return  [file tail [lib_cell $module]]
}

#****************************************************************
# get dict_pin path
#****************************************************************
proc lib_pin { module } {
  return  [lib_dict_value dict_pin $module PATH]
}

#****************************************************************
# get dict_hier path
#****************************************************************
proc lib_hier { module } {
  return  [lib_dict_value dict_hier $module PATH]
}

#****************************************************************
# get dict_src ref
#****************************************************************
proc lib_ref { key } {
  return  [lib_dict_value dict_src $key REF]
}

#****************************************************************
# get dict_sys value
#****************************************************************
proc lib_sys { key } {
  return  [lib_dict_value dict_sys $key]
}

#****************************************************************
# get dict_param value
#****************************************************************
proc lib_param { key } {
  return  [lib_dict_value dict_param $key]
}

#****************************************************************
# get dict_verreg value
#****************************************************************
proc lib_info { args } {
  return  [lib_dict_value dict_verreg {*}$args]
}

#****************************************************************
# sorted dict
#****************************************************************
proc lib_dict_sort { dict_in subkey_list {dir ASCEND} } {
  set list_in {}
  dict for { key val } $dict_in {
    set list_sub {}
    foreach {subkey} $subkey_list {
      lappend list_sub [dict get $val $subkey]
    }
    lappend list_sub $key
    lappend list_in $list_sub
  }
  switch -exact -- $dir {
    {DESCEND} {set lsort_dir -decreasing}
    {ASCEND}  {set lsort_dir -increasing}
    default   {lib_puts ERR "Incorrect sorting direction, should be ASCEND or DESCEND!"}
  }
  foreach {val} [lsort -real -command lib_dict_compare $lsort_dir $list_in] {
    dict set dict_sorted [lindex $val end] {}
  }
  return [dict merge $dict_sorted $dict_in]
}

#****************************************************************
# lib_dict_compare
#****************************************************************
proc lib_dict_compare {a b} {
  set a0 [lindex $a 0]
  set b0 [lindex $b 0]
        if  { $a0 < $b0 } { return -1
  } elseif  { $a0 > $b0 } { return  1
  }
  return [lib_dict_compare [lrange $a 1 end] [lrange $b 1 end]]
}

#****************************************************************
# get dpu size
#****************************************************************
proc lib_dpu_size { arch } {
  set status [scan $arch "B%d" size]
  expr {($status=={0})?{0}:$size}
}

#****************************************************************
# get dpu conv mult
#****************************************************************
proc lib_dpu_mult { arch } {
  expr [lib_dpu_size $arch]/2
}

#****************************************************************
# get divisor
#****************************************************************
proc lib_dpu_divisor { dividend } {
  set divisor {1}
  for {set i 2} {$i<=[expr $dividend/2]} {incr i} {
    if { [expr [expr $dividend/$i]*$i] == $dividend } {
      lappend divisor $i
    }
  }
  lappend divisor $dividend
}

#****************************************************************
# wrap dpu infos
#****************************************************************
proc lib_dpu_infos { dict_verreg } {
  #****************************************************************
  # get info
  #****************************************************************
  set pack_ena          [lib_info info_sys DPU_PACK_ENA       ]
  set dpu_dsp48_ver     [dict get $dict_verreg info_sys DPU_DSP48_VER ]

  #****************************************************************
  # set info
  #****************************************************************
  if { $pack_ena != {1} } {
  dict set dict_verreg info_sys AWRLEN_BW     [expr {($dpu_dsp48_ver=={DSP48E1}?{4}:{8})}]
  dict set dict_verreg info_sys SAXI_ID_BW    [expr {($dpu_dsp48_ver=={DSP48E1}?{12}:{16})}]
  dict set dict_verreg info_sys APB_ADDR_LIB  [lib_dpu_apb_addr_lib   $dict_verreg]
  dict set dict_verreg info_sys APB_IP_LIST   [lib_dpu_apb_ip_list    $dict_verreg]
  dict set dict_verreg info_sys APB_ADDR_LIST [lib_dpu_apb_addr_list  $dict_verreg]
  dict set dict_verreg info_ip                [lib_dpu_ldp_info       $dict_verreg]
  } else {
    dict set dict_verreg info_ip              [lib_dpu_ldp_info_pack  $dict_verreg]
  }
  dict set dict_verreg info_sys GP_INFO       [lib_dpu_gp_info        $dict_verreg]
  dict set dict_verreg info_sys HP_INFO       [lib_dpu_hp_info        $dict_verreg]
  dict set dict_verreg info_sys PS_INFO       [lib_dpu_ps_info        $dict_verreg]
  dict set dict_verreg info_sys IRQ_INFO      [lib_dpu_irq_info       $dict_verreg]

  #****************************************************************
  # return info
  #****************************************************************
  return $dict_verreg
}

#****************************************************************
# info_si_list
#****************************************************************
proc lib_dpu_ghp_info_si_list { dict_verreg m_axi_ghp } {
  set dpu_num       [lib_info info_sys DPU_NUM            ]
  set pack_ena      [lib_info info_sys DPU_PACK_ENA       ]
  set si_list {}
  set bt1120_key  {}
  dict for { ip_name ip_info } [dict get $dict_verreg info_ip] {
    if { $pack_ena != {1} } {
      dict for { pin_dst pin_src } [dict get $ip_info $m_axi_ghp] {
        dict set si_list [string toupper $ip_name]\_$pin_dst IP  $ip_name
        dict set si_list [string toupper $ip_name]\_$pin_dst DST $pin_dst
        if { $ip_name == {bt1120} } {
          set bt1120_key $ip_name\_$pin_dst
        }
      }
    } else {
      for {set dpu_i 0} {$dpu_i<$dpu_num} {incr dpu_i} {
        dict for { pin_dst pin_src } [dict get $ip_info $m_axi_ghp] {
          if { [string match "*_M_AXI*" $pin_dst] != {1} } {
            dict set si_list [string toupper $ip_name]$dpu_i\_$pin_dst IP  $ip_name
            dict set si_list [string toupper $ip_name]$dpu_i\_$pin_dst DST $pin_dst
          } elseif { $dpu_i == [expr $dpu_num -1] } {
            dict set si_list $pin_dst IP  $ip_name
            dict set si_list $pin_dst DST $pin_dst
          }
        }
      }
    }
  }
  if { $bt1120_key != {} } {
    set si_list [dict merge [dict create $bt1120_key {}] $si_list]
  }
  return $si_list
}

#****************************************************************
# info_cn_list
#****************************************************************
proc lib_dpu_ghp_info_cn_list { dict_verreg m_axi_ghp si_list cn_stgy } {
  set bt1120_protect  [expr {($m_axi_ghp=={M_AXI_HP})?[dict exists $dict_verreg info_ip bt1120] : {0} } ]
  set cn_list {}
  foreach pin_dst [dict get $dict_verreg info_sys $m_axi_ghp] {
    dict set cn_list [join "M_AXI_ [string trimleft $pin_dst "S_AXI"] " {}] {}
  }
  set si_num [dict size $si_list]
  set mi_num [llength [dict keys $cn_list]]
  set si_i   {0}
  dict for { si si_info } $si_list {
    if { $si_i < $mi_num } {
      set div_rem [expr $si_i % $mi_num]
      dict lappend cn_list [lindex [dict keys $cn_list] [expr       0+0+$div_rem] ] $si
    } else {
      switch $cn_stgy {
        "stack" {
                  set div_rem 0
                  dict lappend cn_list [lindex [dict keys $cn_list] [expr $mi_num-1-$div_rem] ] $si
                }
        "flat"  -
        default {
                  set div_rem [expr $si_i % ($mi_num - $bt1120_protect)]
                  dict lappend cn_list [lindex [dict keys $cn_list] [expr $mi_num-1-$div_rem] ] $si
                }
      }
    }
    incr si_i
  }
  dict for { mi si_list } $cn_list {
    if { [llength $si_list] == {0} } {
      dict unset cn_list $mi
    }
  }
  return $cn_list
}

#****************************************************************
# info_id_bw_list
#****************************************************************
proc lib_dpu_ghp_info_id_bw_list { si_list cn_list } {
  set si_cn_list [dict values $cn_list]
  foreach si_cn $si_cn_list {
          if { [llength $si_cn] == {0} } {
    } elseif { [llength $si_cn] == {1} } {
      dict set si_list $si_cn ID_BW {6}
    } else {
      set id_bw [expr 6-[expr int(ceil(log([llength $si_cn])/log(2))) ] ]
      foreach si_cn_sub $si_cn {
        dict set si_list $si_cn_sub ID_BW $id_bw
      }
    }
  }
  return $si_list
}

#****************************************************************
# gp_info
#****************************************************************
proc lib_dpu_gp_info { dict_verreg } {
  set cn_stgy [lib_info info_sys M_AXI_CN_STGY]
  if { $cn_stgy == {-1} } {
    set cn_stgy "flat"
  }
  set si_list [lib_dpu_ghp_info_si_list $dict_verreg M_AXI_GP]
  set cn_list [lib_dpu_ghp_info_cn_list $dict_verreg M_AXI_GP $si_list $cn_stgy]
  set si_list [lib_dpu_ghp_info_id_bw_list $si_list $cn_list]
  set gp_info {}
  dict set gp_info GP_SI_LIST $si_list
  dict set gp_info GP_CN_LIST $cn_list
  return $gp_info
}

#****************************************************************
# hp_info
#****************************************************************
proc lib_dpu_hp_info { dict_verreg } {
  set cn_stgy [lib_info info_sys M_AXI_CN_STGY]
  if { $cn_stgy == {-1} } {
    set cn_stgy "flat"
  }
  set si_list [lib_dpu_ghp_info_si_list $dict_verreg M_AXI_HP]
  set cn_list [lib_dpu_ghp_info_cn_list $dict_verreg M_AXI_HP $si_list $cn_stgy]
  set si_list [lib_dpu_ghp_info_id_bw_list $si_list $cn_list]
  set hp_info {}
  dict set hp_info HP_SI_LIST $si_list
  dict set hp_info HP_CN_LIST $cn_list
  return $hp_info
}

#****************************************************************
# info_bw_list
#****************************************************************
proc lib_dpu_ps_info_bw_list { dict_verreg si_list cn_list mi_list} {
  dict for { pin_dst pin_info } $mi_list {
    set ps_src_name [dict get $pin_info SRC_NAME]
    if { [dict exists $cn_list $ps_src_name] } {
      set bw_list {}
      foreach cn_si [dict get $cn_list $ps_src_name] {
        set ip [dict get $si_list $cn_si IP]
        lappend bw_list [dict get $dict_verreg info_ip $ip HP_DATA_BW]
      }
      dict set mi_list $pin_dst BW [lindex [lsort -integer $bw_list] end]
    } else {
      dict set mi_list $pin_dst BW {32}
    }
  }
  return $mi_list
}

#****************************************************************
# ps_info
#****************************************************************
#****************************************************************
# set ps
#   for PSU__CRL_APB__PL?_REF_CTRL__SRCSEL: DPLL, IOPLL, RPLL
#   for GP ports
#       PCW_USE_M_AXI_GP0   :M_AXI_GP0
#       PCW_USE_M_AXI_GP1   :M_AXI_GP1
#       PCW_USE_S_AXI_GP0   :S_AXI_GP0
#       PCW_USE_S_AXI_GP1   :S_AXI_GP1
#       PCW_USE_S_AXI_HP0   :S_AXI_HP0
#       PCW_USE_S_AXI_HP1   :S_AXI_HP1
#       PCW_USE_S_AXI_HP2   :S_AXI_HP2
#       PCW_USE_S_AXI_HP3   :S_AXI_HP3
#       PSU__USE__M_AXI_GP0 :M_AXI_HPM0_FPD
#       PSU__USE__M_AXI_GP1 :M_AXI_HPM1_FPD
#       PSU__USE__M_AXI_GP2 :M_AXI_HPM0_LPD
#       PSU__USE__S_AXI_GP0 :S_AXI_HPC0_FPD
#       PSU__USE__S_AXI_GP1 :S_AXI_HPC1_FPD
#       PSU__USE__S_AXI_GP2 :S_AXI_HP0_FPD
#       PSU__USE__S_AXI_GP3 :S_AXI_HP1_FPD
#       PSU__USE__S_AXI_GP4 :S_AXI_HP2_FPD
#       PSU__USE__S_AXI_GP5 :S_AXI_HP3_FPD
#       PSU__USE__S_AXI_GP6 :S_AXI_LPD
#       PSU__USE__S_AXI_ACP :S_AXI_ACP_FPD
#       PSU__USE__S_AXI_ACE :S_AXI_ACE_FPD
#
#****************************************************************
proc lib_dpu_ps_info { dict_verreg } {
  set dpu_dsp48_ver [dict get $dict_verreg info_sys DPU_DSP48_VER       ]
  set gp_cn_list    [dict get $dict_verreg info_sys GP_INFO GP_CN_LIST  ]
  set hp_si_list    [dict get $dict_verreg info_sys HP_INFO HP_SI_LIST  ]
  set hp_cn_list    [dict get $dict_verreg info_sys HP_INFO HP_CN_LIST  ]
  set ps_ghp_list   {}
  dict for { pin_src pin_info } [concat $gp_cn_list $hp_cn_list] {
    set pin_dst [join "S_AXI_ [string trimleft $pin_src "M_AXI"] " {}]
    switch -exact -- $dpu_dsp48_ver {
      {DSP48E1} {
          switch -exact -- $pin_dst {
            S_AXI_GP0       {set pin_name PCW_USE_S_AXI_GP0   ; set pin_bw {-1}                     ; set pin_clk S_AXI_GP0_ACLK  }
            S_AXI_GP1       {set pin_name PCW_USE_S_AXI_GP1   ; set pin_bw {-1}                     ; set pin_clk S_AXI_GP1_ACLK  }
            S_AXI_HP0       {set pin_name PCW_USE_S_AXI_HP0   ; set pin_bw PCW_S_AXI_HP0_DATA_WIDTH ; set pin_clk S_AXI_HP0_ACLK  }
            S_AXI_HP1       {set pin_name PCW_USE_S_AXI_HP1   ; set pin_bw PCW_S_AXI_HP1_DATA_WIDTH ; set pin_clk S_AXI_HP1_ACLK  }
            S_AXI_HP2       {set pin_name PCW_USE_S_AXI_HP2   ; set pin_bw PCW_S_AXI_HP2_DATA_WIDTH ; set pin_clk S_AXI_HP2_ACLK  }
            S_AXI_HP3       {set pin_name PCW_USE_S_AXI_HP3   ; set pin_bw PCW_S_AXI_HP3_DATA_WIDTH ; set pin_clk S_AXI_HP3_ACLK  }
            default         {lib_error PS_INFO "PS PORT == $pin_dst is not supported..."}
          }
        }
      {DSP48E2} -
      default   {
          switch -exact -- $pin_dst {
            S_AXI_HPC0_FPD  {set pin_name PSU__USE__S_AXI_GP0 ; set pin_bw PSU__SAXIGP0__DATA_WIDTH ; set pin_clk saxihpc0_fpd_aclk }
            S_AXI_HPC1_FPD  {set pin_name PSU__USE__S_AXI_GP1 ; set pin_bw PSU__SAXIGP1__DATA_WIDTH ; set pin_clk saxihpc1_fpd_aclk }
            S_AXI_HP0_FPD   {set pin_name PSU__USE__S_AXI_GP2 ; set pin_bw PSU__SAXIGP2__DATA_WIDTH ; set pin_clk saxihp0_fpd_aclk  }
            S_AXI_HP1_FPD   {set pin_name PSU__USE__S_AXI_GP3 ; set pin_bw PSU__SAXIGP3__DATA_WIDTH ; set pin_clk saxihp1_fpd_aclk  }
            S_AXI_HP2_FPD   {set pin_name PSU__USE__S_AXI_GP4 ; set pin_bw PSU__SAXIGP4__DATA_WIDTH ; set pin_clk saxihp2_fpd_aclk  }
            S_AXI_HP3_FPD   {set pin_name PSU__USE__S_AXI_GP5 ; set pin_bw PSU__SAXIGP5__DATA_WIDTH ; set pin_clk saxihp3_fpd_aclk  }
            S_AXI_LPD       {set pin_name PSU__USE__S_AXI_GP6 ; set pin_bw PSU__SAXIGP6__DATA_WIDTH ; set pin_clk saxi_lpd_aclk     }
            S_AXI_ACP_FPD   {set pin_name PSU__USE__S_AXI_ACP ; set pin_bw {-1}                     ; set pin_clk saxiacp_fpd_aclk  }
            S_AXI_ACE_FPD   {set pin_name PSU__USE__S_AXI_ACE ; set pin_bw {-1}                     ; set pin_clk sacefpd_aclk      }
            default         {lib_error PS_INFO "PS PORT == $pin_dst is not supported..."}
          }
        }
    }
    dict set ps_ghp_list $pin_dst PIN_NAME  $pin_name
    dict set ps_ghp_list $pin_dst PIN_BW    $pin_bw
    dict set ps_ghp_list $pin_dst PIN_CLK   $pin_clk
    dict set ps_ghp_list $pin_dst SRC_NAME  [join "M_AXI_ [string trimleft $pin_dst "S_AXI"] " {}]
    dict set ps_ghp_list $pin_dst SRC_TYPE  [expr {[string match "*bt1120*" $pin_info]?{bt1120}:{dpu} }]
  }
  set ps_ghp_list [lib_dict_sort $ps_ghp_list {} ]
  set ps_ghp_list [lib_dpu_ps_info_bw_list $dict_verreg $hp_si_list $hp_cn_list $ps_ghp_list]
  set ghp_mi_list [lib_dict_sort $ps_ghp_list PIN_NAME ]

  set ps_info {}
  dict set ps_info PS_GHP_LIST  $ps_ghp_list
  dict set ps_info GHP_MI_LIST  $ghp_mi_list
  return $ps_info
}

#****************************************************************
# irq_lib
#****************************************************************
proc lib_dpu_irq_info_lib { dict_verreg } {
  set dpu_dsp48_ver [dict get $dict_verreg info_sys DPU_DSP48_VER ]
  set irq_mod       [dict get $dict_verreg info_sys IRQ_MOD       ]
  set irq_base      {0}
  set irq_lib       {}
  switch -exact -- $dpu_dsp48_ver {
    {DSP48E1}  {
        set irq_base              {0}
        dict set irq_lib dpu      [expr $irq_base+0]
        dict set irq_lib dpu0     [expr $irq_base+0]
        dict set irq_lib dpu1     [expr $irq_base+1]
        dict set irq_lib bt1120   [expr $irq_base+2]
        dict set irq_lib softmax  [expr $irq_base+3]
        dict set irq_lib sigmoid  [expr $irq_base+4]
        dict set irq_lib resize   [expr $irq_base+5]
        dict set irq_lib yrr      [expr $irq_base+6]
      }
    {DSP48E2}  {
        set irq_base              {8}
        dict set irq_lib rgb2yuv  [expr $irq_base+0]
        dict set irq_lib yuv2rgb  [expr $irq_base+1]
        dict set irq_lib resize   [expr $irq_base+1]
        dict set irq_lib yrr      [expr $irq_base+1]
        dict set irq_lib dpu      [expr $irq_base+2]
        dict set irq_lib dpu0     [expr $irq_base+2]
        dict set irq_lib dpu1     [expr $irq_base+3]
        dict set irq_lib dpu2     [expr $irq_base+4]
        dict set irq_lib bt1120   [expr $irq_base+5]
        dict set irq_lib softmax  [expr $irq_base+6]
        dict set irq_lib sigmoid  [expr $irq_base+7]
      }
    default {
        lib_error IRQ_LIB "DPU_DSP48_VER == $dpu_dsp48_ver is not supported..."
      }
  }
  set irq_lib [dict merge $irq_lib $irq_mod]
  return $irq_lib
}

#****************************************************************
# irq_list
#****************************************************************
proc lib_dpu_irq_info_list { dict_verreg irq_lib } {
  set dpu_num       [lib_info info_sys DPU_NUM            ]
  set pack_ena      [lib_info info_sys DPU_PACK_ENA       ]
  set dpu_dsp48_ver [dict get $dict_verreg info_sys DPU_DSP48_VER ]
  set ip_list       [dict keys [dict get $dict_verreg info_ip]    ]
  set irq_min       [expr {($dpu_dsp48_ver=={DSP48E1}?{0}:{8})}   ]
  set irq_max       {16}
  set irq_list      {}
  for {set irq_i $irq_min} {$irq_i<$irq_max} {incr irq_i} {
    dict set irq_list $irq_i IP {c1b0}
    dict set irq_list $irq_i BW {1}
  }
  foreach ip $ip_list {
    dict set irq_list [dict get $irq_lib $ip] IP $ip
    if { $pack_ena == {1} && $dpu_num > {1} } {
      for {set dpu_i 1} {$dpu_i<$dpu_num} {incr dpu_i} {
        dict set irq_list [expr [dict get $irq_lib $ip]+$dpu_i] IP $ip$dpu_i
      }
    }
    if { [lib_info info_ip dpu PROP SFM_ENA] == {1} } {
      dict set irq_list [dict get $irq_lib softmax] IP softmax
    }
  }
  for {set irq_i [expr $irq_max-1]} {$irq_i>=$irq_min} {incr irq_i -1} {
    dict with irq_list $irq_i {
      if { $IP == {c1b0} } {
        dict unset irq_list $irq_i
      } else {
        break
      }
    }
  }
  return $irq_list
}

#****************************************************************
# irq_concat
#****************************************************************
proc lib_dpu_irq_info_concat { irq_list } {
  set irq_i     {0}
  set irq_num   {0}
  set irq_dpu_0 {0}
  set irq_dpu_n {0}
  dict for { irq irq_info } $irq_list {
    incr irq_i
    dict with irq_info {
      if { [string match "dpu*" $IP] } {
        if { $IP == {dpu0} || $IP == {dpu} } {
          set irq_dpu_0 $irq
          incr irq_num
        } else {
          dict unset irq_list $irq
        }
        dict set irq_list $irq_dpu_0 BW [expr $irq +1 - $irq_dpu_0]
      } else {
        incr irq_num 
      }
    }
  }
  return $irq_list
}

#****************************************************************
# irq_info
#****************************************************************
proc lib_dpu_irq_info { dict_verreg } {
  set irq_lib   [lib_dpu_irq_info_lib    $dict_verreg           ]
  set irq_list  [lib_dpu_irq_info_list   $dict_verreg $irq_lib  ]
  set irq_list  [lib_dpu_irq_info_concat $irq_list              ]
  set irq_info  {}
  dict set irq_info IRQ_LIB   $irq_lib
  dict set irq_info IRQ_LIST  $irq_list
  return $irq_info
}

#****************************************************************
# load_parallel_info_pack
#   the key in the M_AXI_HP indicates the interface name
#   if the name has a prefix before M_AXI, it will only be generated once
#****************************************************************
proc lib_dpu_ldp_info_pack { dict_verreg } {
  set info_ip [dict get $dict_verreg info_ip]
  dict for { ip_name ip_dict } $info_ip {
    if { [string match *dpu* $ip_name] } {
      set ldp {2}
      if { [dict exists $ip_dict PROP LOAD_PARALLEL] } {
        set ldp [dict get $ip_dict PROP LOAD_PARALLEL]
      }
      for {set hp_i 0} {$hp_i<$ldp} {incr hp_i} {
        dict set info_ip $ip_name PROP_AUTO "HP$hp_i\_ID_BW" [dict create PROP "HP$hp_i\_ID_BW" PIN_KEY "M_AXI_DATA$hp_i" ]
        dict set info_ip $ip_name M_AXI_HP  "M_AXI_DATA$hp_i" "M_AXI_DATA$hp_i"
      }
      if { [lib_info info_ip $ip_name PROP SFM_ENA] == {1} } {
        dict set info_ip $ip_name M_AXI_HP  "SFM_M_AXI" "SFM_M_AXI"
      }
    }
  }
  return $info_ip
}

#****************************************************************
# wrap dpu ips pack
#****************************************************************
proc lib_dpu_ips_pack { dict_verreg dict_ip } {
  set hier_path   [dict get $dict_verreg info_sys HIER_PATH_DPU ]
  set ip_name     dpu
  set d_ip_name   d_ip_dpu
  dict set dict_ip $d_ip_name [dict create    \
    PATH  $hier_path/dpu                      \
    NAME  [dict get $dict_verreg info_ip $ip_name NAME]  \
    VLNV  {}                                  \
    ]
  if { [dict exists $dict_verreg info_ip $ip_name PROP] } {
    dict for { prop val } [dict get $dict_verreg info_ip $ip_name PROP] {
      dict set dict_ip $d_ip_name PROP $prop $val
    }
  }
  return $dict_ip
}

#****************************************************************
# wrap dpu const
#****************************************************************
proc lib_dpu_ip_const { hier_path c_bw c_val } {
  set ip  [dict create          \
    PATH  $hier_path/[join "dpu_const_ $c_bw b $c_val" {}]  \
    NAME  {xlconstant}          \
    VLNV  {}                    \
    PROP  [dict create          \
          "CONST_WIDTH" $c_bw   \
          "CONST_VAL"   $c_val  \
          ]                     \
    ]
  return $ip
}

#****************************************************************
# wrap dpu ips clk
#****************************************************************
proc lib_dpu_ips_clk { dict_verreg dict_ip } {
  #****************************************************************
  # get info
  #****************************************************************
  set hier_path     [dict get $dict_verreg info_sys HIER_PATH_CLK     ]
  set eu_ena        [dict get $dict_verreg info_sys DPU_EU_ENA        ]
  set lp_ena        [lib_info info_sys DPU_DSP48_LP_ENA   ]
  set dpu_dsp48_ver [dict get $dict_verreg info_sys DPU_DSP48_VER     ]
  set clk_1x_name   {clk_dpu}
  set clk_2x_name   {clk_dsp}
  set clk_1x_freq   [dict get $dict_verreg info_sys DPU_CLK_MHz       ]
  set clk_2x_freq   [expr $clk_1x_freq *2                             ]
  set pack_ena      [lib_info info_sys DPU_PACK_ENA       ]
  set dpu_num       [lib_info info_sys DPU_NUM            ]
		
  #****************************************************************
  # add clk_wiz
  #****************************************************************
  dict set dict_ip d_ip_clk_wiz [dict create  \
    PATH  $hier_path/dpu_clk_wiz              \
    NAME  {clk_wiz}                           \
    VLNV  {}                                  \
    PROP  [dict create                        \
          "RESET_TYPE"  {ACTIVE_LOW}          \
          ]                                   \
    ]
  dict set dict_ip d_ip_clk_wiz PROP "USE_LOCKED"                       {true}
  dict set dict_ip d_ip_clk_wiz PROP "USE_RESET"                        {true}
  dict set dict_ip d_ip_clk_wiz PROP "RESET_TYPE"                       {ACTIVE_LOW}
  if { $eu_ena == {0} } {
    dict set dict_ip d_ip_clk_wiz PROP "CLK_OUT1_PORT"                 $clk_1x_name
    dict set dict_ip d_ip_clk_wiz PROP "CLKOUT1_REQUESTED_OUT_FREQ"    $clk_1x_freq
  } else {
    dict set dict_ip d_ip_clk_wiz PROP "CLK_OUT1_PORT"                 $clk_2x_name
    dict set dict_ip d_ip_clk_wiz PROP "CLKOUT1_REQUESTED_OUT_FREQ"    $clk_2x_freq
    if { $lp_ena != {1} } {
      dict set dict_ip d_ip_clk_wiz PROP "CLKOUT2_USED"                {true}
      dict set dict_ip d_ip_clk_wiz PROP "CLK_OUT2_PORT"               $clk_1x_name
      dict set dict_ip d_ip_clk_wiz PROP "CLKOUT2_REQUESTED_OUT_FREQ"  $clk_1x_freq
      if { $dpu_dsp48_ver == {DSP48E2} } {
        dict set dict_ip d_ip_clk_wiz PROP "PRIMITIVE"                  {Auto}
        dict set dict_ip d_ip_clk_wiz PROP "CLKOUT1_MATCHED_ROUTING"    {true}
        dict set dict_ip d_ip_clk_wiz PROP "CLKOUT2_MATCHED_ROUTING"    {true}
      }
    } else {
      if { $pack_ena != {1} } {
        dict set dict_ip d_ip_clk_wiz PROP "CLKOUT1_DRIVES"             {No_buffer}
      } else {
        # pack & lp
        if { $dpu_dsp48_ver == {DSP48E2} } {
          dict set dict_ip d_ip_clk_wiz PROP "PRIMITIVE"                        {Auto}
          dict set dict_ip d_ip_clk_wiz PROP "CLKOUT1_DRIVES"                   {Buffer_with_CE}
          dict set dict_ip d_ip_clk_wiz PROP "CLKOUT1_MATCHED_ROUTING"          {true}
        }
        for { set i 2 } { $i<=$dpu_num } { incr i } {
          dict set dict_ip d_ip_clk_wiz PROP "CLKOUT$i\_USED"                   {true}
          dict set dict_ip d_ip_clk_wiz PROP "CLK_OUT$i\_PORT"                  $clk_2x_name[expr $i-1]
          dict set dict_ip d_ip_clk_wiz PROP "CLKOUT$i\_REQUESTED_OUT_FREQ"     $clk_2x_freq
          if { $dpu_dsp48_ver == {DSP48E2} } {
            dict set dict_ip d_ip_clk_wiz PROP "CLKOUT$i\_DRIVES"               {Buffer_with_CE}
            dict set dict_ip d_ip_clk_wiz PROP "CLKOUT$i\_MATCHED_ROUTING"      {true}
          }
        }
        set loc_1x  [expr $dpu_num+1]
        dict set dict_ip d_ip_clk_wiz PROP "CLKOUT$loc_1x\_USED"                {true}
        dict set dict_ip d_ip_clk_wiz PROP "CLK_OUT$loc_1x\_PORT"               $clk_1x_name
        dict set dict_ip d_ip_clk_wiz PROP "CLKOUT$loc_1x\_REQUESTED_OUT_FREQ"  $clk_1x_freq
        if { $dpu_dsp48_ver == {DSP48E2} } {
          dict set dict_ip d_ip_clk_wiz PROP "CLKOUT$loc_1x\_MATCHED_ROUTING"   {true}
        }
      }
    }
  }

  #****************************************************************
  # set bufg
  #****************************************************************
  if { $lp_ena == {1} && $pack_ena != {1} } {
    dict set dict_ip d_ip_clk_bufg_dpu [dict create   \
      PATH  $hier_path/dpu_clk_wiz_bufg               \
      REF   [lib_ref sf_bufg_dpu]                     \
      PROP  [dict create                              \
            "DIV"       {2}                           \
            ]                                         \
      ]
  }

  #****************************************************************
  # set reset
  #****************************************************************
  dict set dict_ip d_ip_clk_rst [dict create  \
    PATH  $hier_path/rst_gen_clk  \
    NAME  {proc_sys_reset}        \
    VLNV  {}                      \
    PROP  {}                      \
    ]
  if { $pack_ena == {1} } {
    dict set dict_ip d_ip_clk_rst_dsp [dict create  \
      PATH  $hier_path/rst_gen_clk_dsp  \
      NAME  {proc_sys_reset}            \
      VLNV  {}                          \
      PROP  {}                          \
      ]
  }

  #****************************************************************
  # return dict_ip
  #****************************************************************
  return $dict_ip
}

#****************************************************************
# wrap dpu ips ghp
#****************************************************************
proc lib_dpu_ips_ghp { dict_verreg dict_ip } {
  #****************************************************************
  # get info
  #****************************************************************
  set hier_path     [dict get $dict_verreg info_sys HIER_PATH_GHP       ]
  set cc_en         [dict get $dict_verreg info_sys HP_CC_EN            ]
  set gp_cn_list    [dict get $dict_verreg info_sys GP_INFO GP_CN_LIST  ]
  set hp_cn_list    [dict get $dict_verreg info_sys HP_INFO HP_CN_LIST  ]

  #****************************************************************
  # ghp
  #****************************************************************
  foreach cn_list [list $gp_cn_list $hp_cn_list] {
    dict for { ghp_dst ghp_srcs } $cn_list {
      set ghp_src_num    [llength $ghp_srcs]
      if { $ghp_src_num > {0} } {
        if { $ghp_src_num == {1} && [string match "*bt1120*" [lindex $ghp_srcs 0]] == {1} } {
        } elseif { $cc_en != {0} || $ghp_src_num > {1} } {
          dict set dict_ip d_ip_intc_$ghp_dst  [dict create  \
            PATH  $hier_path/dpu_intc_$ghp_dst \
            NAME  {axi_interconnect}          \
            VLNV	{}                          \
            PROP  [dict create                \
                  "NUM_SI"  $ghp_src_num      \
                  "NUM_MI"  {1}               \
                  ]
            ]
        }
      }
    }
  }
  
  #****************************************************************
  # return dict_ip
  #****************************************************************
  return $dict_ip
}

#****************************************************************
# wrap dpu ips ps
#****************************************************************
proc lib_dpu_ips_ps { dict_verreg dict_ip } {
  set dict_ip_ps  [dict get $dict_verreg info_sys DICT_IP_PS            ]
  set ps_ghp_list [dict get $dict_verreg info_sys PS_INFO PS_GHP_LIST   ]
  dict for { pin_dst pin_info } [lib_dict_sort $ps_ghp_list {}] {
    dict with pin_info {
      dict set dict_ip $dict_ip_ps PROP $PIN_NAME {1}
      if { $PIN_BW != {-1} } {
        dict set dict_ip $dict_ip_ps PROP $PIN_BW   $BW
      }
    }
  }
  return $dict_ip
}

#****************************************************************
# wrap dpu ips irq
#****************************************************************
proc lib_dpu_ips_irq { dict_verreg dict_ip } {
  set hier_path [dict get $dict_verreg info_sys HIER_PATH_IRQ     ]
  set irq_list  [dict get $dict_verreg info_sys IRQ_INFO IRQ_LIST ]
  set num_ports [dict size $irq_list]
  dict set dict_ip d_ip_irq_concat_inner [dict create  \
    PATH  $hier_path/dpu_concat_irq_inner \
    NAME  {xlconcat}                  \
    VLNV  {}                          \
    PROP  [dict create                \
          "NUM_PORTS"   $num_ports    \
          ]                           \
    ]
  set irq_i     {0}
  set need_c1b0 {0}
  dict for { irq irq_info } $irq_list {
    dict with irq_info {
      if { $BW > {1} } {
        dict set dict_ip d_ip_irq_concat_inner PROP "IN$irq_i\_WIDTH.VALUE_SRC"  {USER}
        dict set dict_ip d_ip_irq_concat_inner PROP "IN$irq_i\_WIDTH"            $BW
      } else {
        set need_c1b0 {1}
      }
    }
    incr irq_i
  }
  if { $need_c1b0 != {0} } {
    dict set dict_ip d_ip_irq_c1b0  [lib_dpu_ip_const $hier_path 1  0 ]
  }
  dict set dict_ip d_ip_irq_concat [dict create  \
    PATH  dpu_concat_irq      \
    NAME  {xlconcat}          \
    VLNV  {}                  \
    PROP  [dict create        \
          "NUM_PORTS"   {1}   \
          ]                   \
    ]
  return $dict_ip
}

#****************************************************************
# create pins
#   the prefix ph for hierarchy, pb for bd, pd for dpu_wrap
#   for CLASS:  PORT, PIN, INTF_PORT, INTF_PIN
#   for INTF
#       for MODE:   Master, Slave, System, MirroredMaster, MirroredSlave, MirroredSystem, Monitor
#   for PIN/PORT
#       for TYPE    : CLK, RST, CE, INTR, DATA, UNDEF(default, =OTHER)
#       for DIR     : I, O, IO
#       for VECTOR  : 0, 1
#       for PROP    : (optional)
#****************************************************************

#****************************************************************
# wrap dpu pins pack
#****************************************************************
proc lib_dpu_pins_pack { dict_verreg dict_pin } {
  set hier_path     [dict get $dict_verreg info_sys HIER_PATH_DPU ]
  set saxiclk_indpd [lib_info info_sys DPU_SAXICLK_INDPD ]

  dict set dict_pin pd_dpu_S_AXI      [dict create  "CLASS" INTF_PIN  "PATH" $hier_path/S_AXI       "MODE" Slave  "VLNV" {xilinx.com:interface:aximm_rtl:1.0} ]
  if { $saxiclk_indpd == {1} } {
    dict set dict_pin pd_dpu_S_AXI_CLK  [dict create  "CLASS" PIN       "PATH" $hier_path/S_AXI_CLK   "TYPE" CLK    "DIR" I "VECTOR" 0  "FROM" 0  "TO" 0  ]
    dict set dict_pin pd_dpu_S_AXI_RSTn [dict create  "CLASS" PIN       "PATH" $hier_path/S_AXI_RSTn  "TYPE" RST    "DIR" I "VECTOR" 0  "FROM" 0  "TO" 0  ]
  }

  return $dict_pin
}

#****************************************************************
# wrap dpu pins clk
#****************************************************************
proc lib_dpu_pins_clk { dict_verreg dict_pin } {
  set hier_path   [lib_info info_sys HIER_PATH_CLK      ]
  set eu_ena      [lib_info info_sys DPU_EU_ENA         ]
  set lp_ena      [lib_info info_sys DPU_DSP48_LP_ENA   ]
  set pack_ena    [lib_info info_sys DPU_PACK_ENA       ]
  set dpu_num     [lib_info info_sys DPU_NUM            ]
  dict set dict_pin pd_clk_CLK          [dict create  "CLASS" PIN "PATH" $hier_path/CLK       "TYPE" CLK  "DIR" I "VECTOR" 0  "FROM" 0  "TO" 0  ]
  dict set dict_pin pd_clk_RSTn         [dict create  "CLASS" PIN "PATH" $hier_path/RSTn      "TYPE" RST  "DIR" I "VECTOR" 0  "FROM" 0  "TO" 0  ]
  if { $eu_ena != {0} } {
    dict set dict_pin pd_clk_DSP_CLK    [dict create  "CLASS" PIN "PATH" $hier_path/DSP_CLK   "TYPE" CLK  "DIR" O "VECTOR" 0  "FROM" 0  "TO" 0  ]
  }
  if { $pack_ena == {1} } {
    if { $lp_ena == {1} } {
      for {set i 1} {$i<$dpu_num} {incr i} {
        dict set dict_pin pd_clk_DSP_CLK$i  [dict create  "CLASS" PIN "PATH" $hier_path/DSP_CLK$i "TYPE" CLK  "DIR" O "VECTOR" 0  "FROM" 0  "TO" 0  ]
      }
    }
    dict set dict_pin pd_clk_RSTn_DSP   [dict create  "CLASS" PIN "PATH" $hier_path/RSTn_DSP  "TYPE" RST  "DIR" O "VECTOR" 0  "FROM" 0  "TO" 0  ]
  }
  dict set dict_pin pd_clk_DPU_CLK      [dict create  "CLASS" PIN "PATH" $hier_path/DPU_CLK   "TYPE" CLK  "DIR" O "VECTOR" 0  "FROM" 0  "TO" 0  ]
  dict set dict_pin pd_clk_LOCKED       [dict create  "CLASS" PIN "PATH" $hier_path/LOCKED    "TYPE" DATA "DIR" O "VECTOR" 0  "FROM" 0  "TO" 0  ]
  dict set dict_pin pd_clk_RSTn_INTC    [dict create  "CLASS" PIN "PATH" $hier_path/RSTn_INTC "TYPE" RST  "DIR" O "VECTOR" 0  "FROM" 0  "TO" 0  ]
  dict set dict_pin pd_clk_RSTn_PERI    [dict create  "CLASS" PIN "PATH" $hier_path/RSTn_PERI "TYPE" RST  "DIR" O "VECTOR" 0  "FROM" 0  "TO" 0  ]
  return $dict_pin
}

#****************************************************************
# wrap dpu pins ghp
#****************************************************************
proc lib_dpu_pins_ghp { dict_verreg dict_pin } {
  #****************************************************************
  # get info
  #****************************************************************
  set hier_path   [dict get $dict_verreg info_sys HIER_PATH_GHP       ]
  set gp_si_list  [dict get $dict_verreg info_sys GP_INFO GP_SI_LIST  ]
  set hp_si_list  [dict get $dict_verreg info_sys HP_INFO HP_SI_LIST  ]
  set ghp_mi_list [dict get $dict_verreg info_sys PS_INFO GHP_MI_LIST ]

  #****************************************************************
  # pin in ghp
  #****************************************************************
  set pin_name_list {}
  foreach si_list [list $gp_si_list $hp_si_list] {
    set pin_name_list [concat $pin_name_list [dict keys $si_list]]
  }
  foreach pin_name [lsort $pin_name_list] {
    dict set dict_pin pd_ghp_$pin_name [dict create  "CLASS" INTF_PIN "PATH" $hier_path/$pin_name "MODE" Slave "VLNV" {xilinx.com:interface:aximm_rtl:1.0} ]
  }

  dict set dict_pin pd_ghp_CLK        [dict create  "CLASS" PIN "PATH" $hier_path/CLK       "TYPE" CLK  "DIR" I "VECTOR" 0  "FROM" 0  "TO" 0  ]
  dict set dict_pin pd_ghp_RSTn_INTC  [dict create  "CLASS" PIN "PATH" $hier_path/RSTn_INTC "TYPE" RST  "DIR" I "VECTOR" 0  "FROM" 0  "TO" 0  ]
  dict set dict_pin pd_ghp_RSTn_PERI  [dict create  "CLASS" PIN "PATH" $hier_path/RSTn_PERI "TYPE" RST  "DIR" I "VECTOR" 0  "FROM" 0  "TO" 0  ]
  dict set dict_pin pd_ghp_GHP_CLK_I  [dict create  "CLASS" PIN "PATH" $hier_path/GHP_CLK_I "TYPE" CLK  "DIR" I "VECTOR" 0  "FROM" 0  "TO" 0  ]
  dict set dict_pin pd_ghp_GHP_RSTn   [dict create  "CLASS" PIN "PATH" $hier_path/GHP_RSTn  "TYPE" RST  "DIR" I "VECTOR" 0  "FROM" 0  "TO" 0  ]
  dict set dict_pin pd_ghp_GHP_CLK_O  [dict create  "CLASS" PIN "PATH" $hier_path/GHP_CLK_O "TYPE" CLK  "DIR" O "VECTOR" 0  "FROM" 0  "TO" 0  ]

  #****************************************************************
  # pin out ghp
  #****************************************************************
  dict for { pin_dst pin_info } $ghp_mi_list {
    set pin_name [dict get $pin_info SRC_NAME]
    dict set dict_pin pd_ghp_$pin_name [dict create  "CLASS" INTF_PIN "PATH" $hier_path/$pin_name "MODE" Master "VLNV" {xilinx.com:interface:aximm_rtl:1.0} ]
  }

  return $dict_pin
}

#****************************************************************
# wrap dpu pins irq
#****************************************************************
proc lib_dpu_pins_irq { dict_verreg dict_pin } {
  #****************************************************************
  # get info
  #****************************************************************
  set hier_path   [dict get $dict_verreg info_sys HIER_PATH_IRQ     ]
  set irq_list    [dict get $dict_verreg info_sys IRQ_INFO IRQ_LIST ]

  #****************************************************************
  # pin out ghp
  #****************************************************************
  if { [dict size $irq_list] != {0} } {
    dict set dict_pin pd_irq_INTR [dict create  "CLASS" PIN "PATH" $hier_path/INTR "TYPE" INTR  "DIR" O "VECTOR" 0  "FROM" 0  "TO" 0  ]
  }
  
  return $dict_pin
}

#****************************************************************
# set connect dict
#   top keys are
#     {PIN from PIN}, {PORT from PIN}, {PIN from PORT}, {PORT from PORT}
#     {PIN intf PIN}, {PORT intf PIN}, {PIN intf PORT}, {PORT intf PORT}
#   the key is dst, the value is src
#****************************************************************

#****************************************************************
# wrap dpu cn pack
#****************************************************************
proc lib_dpu_cns_pack { dict_verreg dict_cn } {

  #****************************************************************
  # get info
  #****************************************************************
  set lp_ena            [lib_info info_sys DPU_DSP48_LP_ENA  ]
  set dpu_num           [lib_info info_sys DPU_NUM           ]
  set saxiclk_indpd     [lib_info info_sys DPU_SAXICLK_INDPD ]

  #****************************************************************
  #
  #****************************************************************
  dict set dict_cn cn_dpu   {PIN intf PIN}  [lib_cell d_ip_dpu]/S_AXI         [lib_pin pd_dpu_S_AXI]
  if { $saxiclk_indpd == {1} } {
    dict set dict_cn cn_dpu {PIN from PIN}  [lib_cell d_ip_dpu]/s_axi_aclk    [lib_pin pd_dpu_S_AXI_CLK]
    dict set dict_cn cn_dpu {PIN from PIN}  [lib_cell d_ip_dpu]/s_axi_aresetn [lib_pin pd_dpu_S_AXI_RSTn]
  }

  dict set dict_cn cn_dpu   {PIN from PIN}  [lib_cell d_ip_dpu]/dpu_2x_clk          [lib_pin pd_clk_DSP_CLK]
  dict set dict_cn cn_dpu   {PIN from PIN}  [lib_cell d_ip_dpu]/m_axi_dpu_aclk      [lib_pin pd_clk_DPU_CLK]
  dict set dict_cn cn_dpu   {PIN from PIN}  [lib_cell d_ip_dpu]/dpu_2x_resetn       [lib_pin pd_clk_RSTn_DSP]
  dict set dict_cn cn_dpu   {PIN from PIN}  [lib_cell d_ip_dpu]/m_axi_dpu_aresetn   [lib_pin pd_clk_RSTn_PERI]
  if { $lp_ena == {1} } {
    for { set i 1 } { $i<$dpu_num } { incr i } {
      dict set dict_cn cn_dpu   {PIN from PIN}  [lib_cell d_ip_dpu]/dpu$i\_2x_clk   [lib_pin pd_clk_DSP_CLK$i]
    }
  }

  #****************************************************************
  # return dict_cn
  #****************************************************************
  return $dict_cn

}

#****************************************************************
# wrap dpu cn pack clk
#****************************************************************
proc lib_dpu_cns_pack_clk { dict_verreg dict_cn } {

  #****************************************************************
  # get info
  #****************************************************************
  set lp_ena        [lib_info info_sys DPU_DSP48_LP_ENA   ]
  set dpu_dsp48_ver [lib_info info_sys DPU_DSP48_VER      ]
  set dpu_num       [lib_info info_sys DPU_NUM            ]

  #****************************************************************
  # pd in
  #****************************************************************
  dict set dict_cn cn_pd_clk_in {PIN from PIN} [lib_cell d_ip_clk_wiz]/clk_in1  [lib_pin pd_clk_CLK ]
  dict set dict_cn cn_pd_clk_in {PIN from PIN} [lib_cell d_ip_clk_wiz]/resetn   [lib_pin pd_clk_RSTn]
  if { $lp_ena == {1} && $dpu_dsp48_ver == {DSP48E2} } {
    dict set dict_cn cn_pd_clk_in {PIN from PIN} [lib_cell d_ip_clk_wiz]/clk_dsp_ce       [lib_cell d_ip_dpu]/dpu_2x_clk_ce
    for { set i 1 } { $i<$dpu_num } { incr i } {
      dict set dict_cn cn_pd_clk_in {PIN from PIN} [lib_cell d_ip_clk_wiz]/clk_dsp$i\_ce  [lib_cell d_ip_dpu]/dpu$i\_2x_clk_ce
    }
  }

  #****************************************************************
  # pd out clk
  #****************************************************************
  dict set dict_cn cn_pd_clk_res {PIN from PIN} [lib_pin pd_clk_DSP_CLK]        [lib_cell d_ip_clk_wiz]/clk_dsp
  dict set dict_cn cn_pd_clk_res {PIN from PIN} [lib_pin pd_clk_DPU_CLK]        [lib_cell d_ip_clk_wiz]/clk_dpu
  if { $lp_ena == {1} } {
    for { set i 1 } { $i<$dpu_num } { incr i } {
      dict set dict_cn cn_pd_clk_res {PIN from PIN} [lib_pin pd_clk_DSP_CLK$i]  [lib_cell d_ip_clk_wiz]/clk_dsp$i
    }
  }

  #****************************************************************
  # pd out locked
  #****************************************************************
  dict set dict_cn cn_pd_clk_res {PIN from PIN} [lib_pin pd_clk_LOCKED]         [lib_cell d_ip_clk_wiz]/locked

  #****************************************************************
  # rst
  #****************************************************************
  dict set dict_cn cn_pd_clk_rst {PIN from PIN} [lib_cell d_ip_clk_rst]/slowest_sync_clk      [lib_cell d_ip_clk_wiz]/clk_dpu
  dict set dict_cn cn_pd_clk_rst {PIN from PIN} [lib_cell d_ip_clk_rst]/ext_reset_in          [lib_pin pd_clk_RSTn]
  dict set dict_cn cn_pd_clk_rst {PIN from PIN} [lib_cell d_ip_clk_rst]/dcm_locked            [lib_cell d_ip_clk_wiz]/locked
  dict set dict_cn cn_pd_clk_rst {PIN from PIN} [lib_cell d_ip_clk_rst_dsp]/slowest_sync_clk  [lib_cell d_ip_clk_wiz]/clk_dsp
  dict set dict_cn cn_pd_clk_rst {PIN from PIN} [lib_cell d_ip_clk_rst_dsp]/ext_reset_in      [lib_pin pd_clk_RSTn]
  dict set dict_cn cn_pd_clk_rst {PIN from PIN} [lib_cell d_ip_clk_rst_dsp]/dcm_locked        [lib_cell d_ip_clk_wiz]/locked

  #****************************************************************
  # pd out rst
  #****************************************************************
  dict set dict_cn cn_pd_clk_res {PIN from PIN} [lib_pin pd_clk_RSTn_DSP]   [lib_cell d_ip_clk_rst_dsp]/peripheral_aresetn
  dict set dict_cn cn_pd_clk_res {PIN from PIN} [lib_pin pd_clk_RSTn_INTC]  [lib_cell d_ip_clk_rst]/interconnect_aresetn
  dict set dict_cn cn_pd_clk_res {PIN from PIN} [lib_pin pd_clk_RSTn_PERI]  [lib_cell d_ip_clk_rst]/peripheral_aresetn

  #****************************************************************
  # return dict_cn
  #****************************************************************
  return $dict_cn

}

#****************************************************************
# wrap dpu cn ghp
#****************************************************************
proc lib_dpu_cns_ghp { dict_verreg dict_cn } {
  #****************************************************************
  # get info
  #****************************************************************
  set pack_ena    [lib_info info_sys DPU_PACK_ENA       ]
  set cc_en       [dict get $dict_verreg info_sys HP_CC_EN            ]
  set gp_si_list  [dict get $dict_verreg info_sys GP_INFO GP_SI_LIST  ]
  set gp_cn_list  [dict get $dict_verreg info_sys GP_INFO GP_CN_LIST  ]
  set hp_si_list  [dict get $dict_verreg info_sys HP_INFO HP_SI_LIST  ]
  set hp_cn_list  [dict get $dict_verreg info_sys HP_INFO HP_CN_LIST  ]
  set ghp_mi_list [dict get $dict_verreg info_sys PS_INFO GHP_MI_LIST ]
  set dict_ip_ps  [dict get $dict_verreg info_sys DICT_IP_PS          ]
  set ps_path     [lib_cell $dict_ip_ps                               ]

  #****************************************************************
  # cn ghp in
  #****************************************************************
  foreach si_list [list $gp_si_list $hp_si_list] {
    dict for { pin_name pin_info } $si_list {
      if { $pack_ena != {1} } { set pin_src [lib_pin pd_dpu_$pin_name]
      } else {                  set pin_src [lib_cell d_ip_dpu]/$pin_name
      }
      dict set dict_cn cn_dpu_ghp_in {PIN intf PIN} [lib_pin pd_ghp_$pin_name] $pin_src
    }
  }
  dict set dict_cn cn_dpu_ghp_in {PIN from PIN} [lib_pin pd_ghp_CLK]         [lib_pin pd_clk_DPU_CLK]
  dict set dict_cn cn_dpu_ghp_in {PIN from PIN} [lib_pin pd_ghp_RSTn_INTC]   [lib_pin pd_clk_RSTn_INTC]
  dict set dict_cn cn_dpu_ghp_in {PIN from PIN} [lib_pin pd_ghp_RSTn_PERI]   [lib_pin pd_clk_RSTn_PERI]
  if { $cc_en == {0} } {
    dict set dict_cn cn_dpu_ghp_in {PIN from PIN} [lib_pin pd_ghp_GHP_CLK_I] [lib_pin pd_clk_DPU_CLK]
    dict set dict_cn cn_dpu_ghp_in {PIN from PIN} [lib_pin pd_ghp_GHP_RSTn]  [lib_pin pd_clk_RSTn_PERI]
  }
  dict set dict_cn cn_dpu_ghp_in {PIN from PIN} [lib_pin pd_ghp_GHP_CLK_O] [lib_pin pd_ghp_GHP_CLK_I]

  #****************************************************************
  # cn ghp
  #****************************************************************
  foreach cn_list [list $gp_cn_list $hp_cn_list] {
    dict for { dst srcs } $cn_list {
      set src_num [llength $srcs]
      if { $src_num > {0} } {
        if { [llength $srcs] == {1} && [string match "*bt1120*" [lindex $srcs 0]] == {1} } {
          dict set dict_cn cn_dpu_ghp_intc {PIN intf PIN} [lib_pin pd_ghp_$dst] [lib_pin pd_ghp_[lindex $srcs 0]]
        } elseif { $cc_en != {0} || $src_num > {1} } {
          dict set dict_cn cn_dpu_ghp_intc {PIN from PIN} [lib_cell d_ip_intc_$dst]/ACLK    [lib_pin pd_ghp_CLK]
          dict set dict_cn cn_dpu_ghp_intc {PIN from PIN} [lib_cell d_ip_intc_$dst]/ARESETN [lib_pin pd_ghp_RSTn_INTC]
          set s_cnt {0}
          foreach pin_name $srcs {
            dict set dict_cn cn_dpu_ghp_intc {PIN intf PIN} [lib_cell d_ip_intc_$dst]/S0$s_cnt\_AXI      [lib_pin pd_ghp_$pin_name]
            dict set dict_cn cn_dpu_ghp_intc {PIN from PIN} [lib_cell d_ip_intc_$dst]/S0$s_cnt\_ACLK     [lib_pin pd_ghp_CLK]
            dict set dict_cn cn_dpu_ghp_intc {PIN from PIN} [lib_cell d_ip_intc_$dst]/S0$s_cnt\_ARESETN  [lib_pin pd_ghp_RSTn_PERI]
            incr s_cnt
          }
          dict set dict_cn cn_dpu_ghp_intc {PIN intf PIN} [lib_cell d_ip_intc_$dst]/M00\_AXI      [lib_pin pd_ghp_$dst]
          dict set dict_cn cn_dpu_ghp_intc {PIN from PIN} [lib_cell d_ip_intc_$dst]/M00\_ACLK     [lib_pin pd_ghp_GHP_CLK_I]
          dict set dict_cn cn_dpu_ghp_intc {PIN from PIN} [lib_cell d_ip_intc_$dst]/M00\_ARESETN  [lib_pin pd_ghp_GHP_RSTn]
        } else {
          dict set dict_cn cn_dpu_ghp_intc {PIN intf PIN} [lib_pin pd_ghp_$dst] [lib_pin pd_ghp_[lindex $srcs 0]] 
        }
      }
    }
  }

  #****************************************************************
  # cn ghp ps
  #****************************************************************
  dict for { pin_dst pin_info } $ghp_mi_list {
    dict with pin_info {
      dict set dict_cn cn_dpu_ghp_ps {PIN intf PIN} $ps_path/$pin_dst [lib_pin pd_ghp_$SRC_NAME]
      switch -exact -- $SRC_TYPE {
        {dpu}     { dict set dict_cn cn_dpu_ghp_ps {PIN from PIN} $ps_path/$PIN_CLK [lib_pin pd_ghp_GHP_CLK_O]  }
        {bt1120}  { dict set dict_cn cn_dpu_ghp_ps {PIN from PIN} $ps_path/$PIN_CLK [lib_pin pd_bt1120_clk]     }
        default   { lib_error DPU "SRC_TYPE == $SRC_TYPE is not supported ..."                                  }
      }
    }
  }

  #****************************************************************
  # return dict_cn
  #****************************************************************
  return $dict_cn

}

#****************************************************************
# wrap dpu cn irq
#****************************************************************
proc lib_dpu_cns_irq { dict_verreg dict_cn } {
  #****************************************************************
  # get info
  #****************************************************************
  set dsp48_ver   [dict get $dict_verreg info_sys DPU_DSP48_VER ]
  set dict_ip_ps  [dict get $dict_verreg info_sys DICT_IP_PS    ]
  set irq_list    [dict get $dict_verreg info_sys IRQ_INFO IRQ_LIST ]
  set pack_ena    [lib_info info_sys DPU_PACK_ENA       ]

  #****************************************************************
  # irq
  #****************************************************************
  set num_ports   [dict size $irq_list]
  set irq_i       {0}
  dict for { irq irq_info } $irq_list {
    dict with irq_info {
      if { $IP == {c1b0} }  {set pin_src [lib_cell d_ip_irq_c1b0]/dout
      } elseif { $pack_ena == {1} } {
        if { $IP == {softmax} } {
                             set pin_src [lib_cell d_ip_dpu]/sfm_interrupt
        } else {
                             set pin_src [lib_cell d_ip_dpu]/dpu_interrupt
        }
      } else                {set pin_src [lib_pin pd_$IP\_INTR]
      }
      dict set dict_cn cn_dpu_irq {PIN from PIN} [lib_cell d_ip_irq_concat_inner]/In$irq_i $pin_src
    }
    incr irq_i
  }

  #****************************************************************
  # ps
  #****************************************************************
  set pin_ps_irq  [lib_cell $dict_ip_ps]/[expr {($dsp48_ver=={DSP48E1}?{IRQ_F2P}:{pl_ps_irq1})}]
  dict set dict_cn cp_dpu_irq {PIN from PIN} [lib_pin pd_irq_INTR] [lib_cell d_ip_irq_concat_inner]/dout
  dict set dict_cn cp_dpu_irq {PIN from PIN} [lib_cell d_ip_irq_concat]/In0 [lib_pin pd_irq_INTR]
  dict set dict_cn cp_dpu_irq {PIN from PIN} $pin_ps_irq [lib_cell d_ip_irq_concat]/dout

  #****************************************************************
  # return dict_cn
  #****************************************************************
  return $dict_cn

}

#****************************************************************
# print log table
#****************************************************************
proc lib_bd_logs { {chan stdout}  } {
  global dict_prj
  if { [lib_sys flow_edt_bd] == 1 || [lib_sys flow_crt_bd] ==1 } {
    lib_puts_logo ENV $chan
  }
  set bd_logs {}
  if { [dict exists $dict_prj dict_ip ps] == {1} } {
    set ip_ps       [lib_cell_name ps] 
    set ip_ps_base  [lib_rela_path [lib_value ps BASE]]
  } else {
    set ip_ps       none
    set ip_ps_base  none
  }
  dict with dict_prj dict_sys {
    set bd_logs [dict create                                \
        "ver_vivado"    $ver_vivado                         \
        "ver_date"      $ver_year.$ver_month.$ver_day       \
        "ver_bit"       $ver_bit                            \
        "pwd_dir"       $pwd_dir                            \
        "prj_name"      $prj_name                           \
        "prj_part"      $prj_part                           \
        "bd_ooc"        $bd_ooc                             \
        "bd_path"       [lib_rela_path $bd_path]            \
        "ip_ps_base"    $ip_ps_base                         \
        ]
  }
  set cnt_sys       [llength [dict keys $bd_logs]]
  if { [dict exists $dict_prj dict_param] == {1} } {
    dict for { param value } [dict get $dict_prj dict_param] {
      dict set bd_logs $param $value
    }
  }
  set col_list      [list NAME VALUE]
  set width_list    [list [lib_llength_max [dict keys $bd_logs]] [lib_llength_max [dict values $bd_logs]]]
  set fmtstr_cont   [lib_tbl_frmstr_content $width_list]
  set fmtstr_boder  [lib_tbl_frmstr_border  $width_list]
  lib_tbl_title_bar ENV $fmtstr_boder $fmtstr_cont $col_list $chan
  set cnt           0
  set cnt_all       [llength [dict keys $bd_logs]]
  dict for { key val } $bd_logs {
    lib_puts ENV [format $fmtstr_cont $key $val] $chan
    incr cnt
    if { ( [format %.0f [expr fmod($cnt,3)]]==0 && $cnt<$cnt_sys )  \
        || $cnt==$cnt_sys || $cnt==$cnt_all } {
      lib_tbl_border ENV $fmtstr_boder $chan
    }
  }
}

#****************************************************************
# create block design
#****************************************************************
proc lib_create_bd {bd_name bd_dir bd_ooc bd_path} {
  lib_puts BLK "Creating block design $bd_name..."
  create_bd_design $bd_name -dir $bd_dir -quiet
  set_property synth_checkpoint_mode $bd_ooc [get_files $bd_path]
  lib_puts BLK "Successfully created block design $bd_name!"
}

#****************************************************************
# open block design
#****************************************************************
proc lib_open_bd {bd_name bd_path} {
  lib_puts BLK "Opening block design $bd_name..."
  open_bd_design $bd_path -quiet
  lib_puts BLK "Successfully opened block design $bd_name!"
}

#****************************************************************
# copy ip repo
#****************************************************************
proc lib_copy_iprepo {  } {
  global dict_prj
  set ip_dir      [lib_sys ip_dir]
  set ip_temp_dir [lib_sys prj_temp_dir]/srcs/ip
  file delete -force $ip_temp_dir
  file mkdir         $ip_temp_dir
  if { [dict exists $dict_prj dict_ip_sel] == {1} } {
    set ip_sel [dict get $dict_prj dict_ip_sel]
  } else {
    set srcs_full [glob -type d -directory [lib_sys ip_dir] *]
    foreach { src_full } $srcs_full {
      dict set ip_sel [lindex [file split $src_full] end] [lindex [file split $src_full] end] 
    }
  }
  lib_puts BLK "Copying ip repository..."
  set col_list    [list DST SRC]
  set width_dst   [lindex $col_list 0]
  set width_src   [lindex $col_list 1]
  dict for { dst src } $ip_sel {
    file copy -force $ip_dir/$src $ip_temp_dir/$dst
    lappend width_dst [lib_rela_path $ip_temp_dir/$dst  ]
    lappend width_src [lib_rela_path $ip_dir/$src       ]
  }
  set width_list    [list [lib_llength_max $width_dst] [lib_llength_max $width_src] ]
  set fmtstr_cont   [lib_tbl_frmstr_content $width_list]
  set fmtstr_boder  [lib_tbl_frmstr_border  $width_list]
  lib_tbl_title_bar BLK $fmtstr_boder $fmtstr_cont $col_list
  for { set i 1 } { $i<[llength $width_dst] } {incr i} {
    lib_puts BLK [format $fmtstr_cont [lindex $width_dst $i] [lindex $width_src $i] ]
  }
  lib_tbl_border BLK $fmtstr_boder
  lib_puts BLK "Successfully copied all ips into [lib_rela_path $ip_temp_dir]"
}

#****************************************************************
# create_hierarchies
#****************************************************************
proc lib_create_hiers { dict_hier } {
  lib_puts  BLK "Creating hierarchies..."
  dict for { hier_name hier_info } $dict_hier {
    set path          [dict get $hier_info PATH]
    create_bd_cell -type hier $path
    lib_puts BLK "Created hierarchy $path!"
  }
  lib_puts BLK "Successfully created all hierarchies!"
}

#****************************************************************
# create_ips
#****************************************************************
proc lib_create_ips { dict_ip } {
  lib_puts  BLK "Creating ips..."
  dict for { ip_name ip_info } $dict_ip {
    # create ip
    set path          [dict get $ip_info PATH]
    
    # 
    if {([dict exists $ip_info REF] != 0) && ([dict get $ip_info REF] != {})} {
      create_bd_cell -type module -reference [dict get $ip_info REF] $path
    } else {
      if {([dict exists $ip_info VLNV] == 0) || ([dict get $ip_info VLNV] == {})} {
        if {([dict exists $ip_info NAME] == 0) || ([dict get $ip_info NAME] == {})} {
          set filter "\{NAME=~$ip_name && UPGRADE_VERSIONS==\"\"\}"
        } else {
          set filter "\{NAME=~[dict get $ip_info NAME] && UPGRADE_VERSIONS==\"\"\}"
        }
        set vlnv [get_ipdefs -all -filter [expr $filter] ]
      } else {
        set vlnv [dict get $ip_info VLNV]
      }
      create_bd_cell -type ip -vlnv $vlnv $path
    }
    # load base tcl
    set dict_prop {}
    if {[dict exists $ip_info BASE]} {
      set base_path [dict get $ip_info BASE]
      #source $base_path
    set dict_prop [dict create                                          \
    CONFIG.PSU__CAN1__PERIPHERAL__ENABLE            {1}             \
    CONFIG.PSU__CAN1__PERIPHERAL__IO                {MIO 24 .. 25}  \
    CONFIG.PSU__DPAUX__PERIPHERAL__IO               {MIO 27 .. 30}  \
    CONFIG.PSU__ENET3__PERIPHERAL__ENABLE           {1}             \
    CONFIG.PSU__ENET3__GRP_MDIO__ENABLE             {1}             \
    CONFIG.PSU__GPIO0_MIO__PERIPHERAL__ENABLE       {1}             \
    CONFIG.PSU__GPIO1_MIO__PERIPHERAL__ENABLE       {1}             \
    CONFIG.PSU__I2C0__PERIPHERAL__ENABLE            {1}             \
    CONFIG.PSU__I2C0__PERIPHERAL__IO                {MIO 14 .. 15}  \
    CONFIG.PSU__I2C1__PERIPHERAL__ENABLE            {1}             \
    CONFIG.PSU__I2C1__PERIPHERAL__IO                {MIO 16 .. 17}  \
    CONFIG.PSU__PCIE__PERIPHERAL__ENABLE            {0}             \
    CONFIG.PSU__PCIE__PERIPHERAL__ROOTPORT_IO       {MIO 31}        \
    CONFIG.PSU__PCIE__DEVICE_PORT_TYPE              {Root Port}     \
    CONFIG.PSU__PCIE__BAR0_ENABLE                   {0}             \
    CONFIG.PSU__PCIE__DEVICE_ID                     {0xD021}        \
    CONFIG.PSU__PCIE__CLASS_CODE_BASE               {0x06}          \
    CONFIG.PSU__PCIE__CLASS_CODE_SUB                {0x04}          \
    CONFIG.PSU__PCIE__CRS_SW_VISIBILITY             {1}             \
    CONFIG.PSU__DP__LANE_SEL                        {Dual Lower}    \
    CONFIG.PSU__PMU__PERIPHERAL__ENABLE             {1}             \
    CONFIG.PSU__PMU__GPI0__ENABLE                   {0}             \
    CONFIG.PSU__PMU__GPI1__ENABLE                   {0}             \
    CONFIG.PSU__PMU__GPI2__ENABLE                   {0}             \
    CONFIG.PSU__PMU__GPI3__ENABLE                   {0}             \
    CONFIG.PSU__PMU__GPI4__ENABLE                   {0}             \
    CONFIG.PSU__PMU__GPI5__ENABLE                   {0}             \
    CONFIG.PSU__PMU__GPO2__POLARITY                 {high}          \
    CONFIG.PSU__QSPI__PERIPHERAL__ENABLE            {1}             \
    CONFIG.PSU__QSPI__PERIPHERAL__MODE              {Dual Parallel} \
    CONFIG.PSU__QSPI__GRP_FBCLK__ENABLE             {1}             \
    CONFIG.PSU__SD1__PERIPHERAL__ENABLE             {1}             \
    CONFIG.PSU__SD1__GRP_CD__ENABLE                 {1}             \
    CONFIG.PSU__SD1__GRP_WP__ENABLE                 {1}             \
    CONFIG.PSU__SWDT0__PERIPHERAL__ENABLE           {1}             \
    CONFIG.PSU__SWDT1__PERIPHERAL__ENABLE           {1}             \
    CONFIG.PSU__TTC0__PERIPHERAL__ENABLE            {1}             \
    CONFIG.PSU__TTC1__PERIPHERAL__ENABLE            {1}             \
    CONFIG.PSU__TTC2__PERIPHERAL__ENABLE            {1}             \
    CONFIG.PSU__TTC3__PERIPHERAL__ENABLE            {1}             \
    CONFIG.PSU__DDRC__BANK_ADDR_COUNT               {2}             \
    CONFIG.PSU__DDRC__BUS_WIDTH                     {64 Bit}        \
    CONFIG.PSU__DDRC__CL                            {15}            \
    CONFIG.PSU__DDRC__CLOCK_STOP_EN                 {0}             \
    CONFIG.PSU_DYNAMIC_DDR_CONFIG_EN                {1}             \
    CONFIG.PSU__DDRC__COL_ADDR_COUNT                {10}            \
    CONFIG.PSU__DDRC__RANK_ADDR_COUNT               {0}             \
    CONFIG.PSU__DDRC__CWL                           {14}            \
    CONFIG.PSU__DDRC__BG_ADDR_COUNT                 {2}             \
    CONFIG.PSU__DDRC__DEVICE_CAPACITY               {4096 MBits}    \
    CONFIG.PSU__DDRC__DRAM_WIDTH                    {8 Bits}        \
    CONFIG.PSU__DDRC__ECC                           {Disabled}      \
    CONFIG.PSU__DDRC__MEMORY_TYPE                   {DDR 4}         \
    CONFIG.PSU__DDRC__ROW_ADDR_COUNT                {15}            \
    CONFIG.PSU__DDRC__SPEED_BIN                     {DDR4_2133P}    \
    CONFIG.PSU__DDRC__T_FAW                         {30.0}          \
    CONFIG.PSU__DDRC__T_RAS_MIN                     {33}            \
    CONFIG.PSU__DDRC__T_RC                          {47.06}         \
    CONFIG.PSU__DDRC__T_RCD                         {15}            \
    CONFIG.PSU__DDRC__T_RP                          {15}            \
    CONFIG.PSU__DDRC__TRAIN_DATA_EYE                {1}             \
    CONFIG.PSU__DDRC__TRAIN_READ_GATE               {1}             \
    CONFIG.PSU__DDRC__TRAIN_WRITE_LEVEL             {1}             \
    CONFIG.PSU__DDRC__VREF                          {1}             \
    CONFIG.PSU__DDRC__BRC_MAPPING                   {ROW_BANK_COL}  \
    CONFIG.PSU__DDRC__DIMM_ADDR_MIRROR              {0}             \
    CONFIG.PSU__DDRC__STATIC_RD_MODE                {0}             \
    CONFIG.PSU__DDRC__DEEP_PWR_DOWN_EN              {0}             \
    CONFIG.PSU__DDRC__DDR4_T_REF_MODE               {0}             \
    CONFIG.PSU__DDRC__DDR4_T_REF_RANGE              {Normal (0-85)} \
    CONFIG.PSU__DDRC__DDR3_T_REF_RANGE              {NA}            \
    CONFIG.PSU__DDRC__DDR3L_T_REF_RANGE             {NA}            \
    CONFIG.PSU__DDRC__LPDDR3_T_REF_RANGE            {NA}            \
    CONFIG.PSU__DDRC__LPDDR4_T_REF_RANGE            {NA}            \
    CONFIG.PSU__DDRC__PHY_DBI_MODE                  {0}             \
    CONFIG.PSU__DDRC__DM_DBI                        {DM_NO_DBI}     \
    CONFIG.PSU__DDRC__COMPONENTS                    {UDIMM}         \
    CONFIG.PSU__DDRC__PARITY_ENABLE                 {0}             \
    CONFIG.PSU__DDRC__DDR4_CAL_MODE_ENABLE          {0}             \
    CONFIG.PSU__DDRC__DDR4_CRC_CONTROL              {0}             \
    CONFIG.PSU__DDRC__FGRM                          {1X}            \
    CONFIG.PSU__DDRC__VENDOR_PART                   {OTHERS}        \
    CONFIG.PSU__DDRC__SB_TARGET                     {15-15-15}      \
    CONFIG.PSU__DDRC__LP_ASR                        {manual normal} \
    CONFIG.PSU__DDRC__DDR4_ADDR_MAPPING             {0}             \
    CONFIG.PSU__DDRC__SELF_REF_ABORT                {0}             \
    CONFIG.PSU__DDRC__ADDR_MIRROR                   {0}             \
    CONFIG.PSU__DDRC__PER_BANK_REFRESH              {0}             \
    CONFIG.PSU__DDRC__ENABLE_LP4_SLOWBOOT           {0}             \
    CONFIG.PSU__DDRC__ENABLE_LP4_HAS_ECC_COMP       {0}             \
    CONFIG.PSU__DDRC__DQMAP_0_3                     {0}             \
    CONFIG.PSU__DDRC__DQMAP_4_7                     {0}             \
    CONFIG.PSU__DDRC__DQMAP_8_11                    {0}             \
    CONFIG.PSU__DDRC__DQMAP_12_15                   {0}             \
    CONFIG.PSU__DDRC__DQMAP_16_19                   {0}             \
    CONFIG.PSU__DDRC__DQMAP_20_23                   {0}             \
    CONFIG.PSU__DDRC__DQMAP_24_27                   {0}             \
    CONFIG.PSU__DDRC__DQMAP_28_31                   {0}             \
    CONFIG.PSU__DDRC__DQMAP_32_35                   {0}             \
    CONFIG.PSU__DDRC__DQMAP_36_39                   {0}             \
    CONFIG.PSU__DDRC__DQMAP_40_43                   {0}             \
    CONFIG.PSU__DDRC__DQMAP_44_47                   {0}             \
    CONFIG.PSU__DDRC__DQMAP_48_51                   {0}             \
    CONFIG.PSU__DDRC__DQMAP_52_55                   {0}             \
    CONFIG.PSU__DDRC__DQMAP_56_59                   {0}             \
    CONFIG.PSU__DDRC__DQMAP_60_63                   {0}             \
    CONFIG.PSU__DDRC__DQMAP_64_67                   {0}             \
    CONFIG.PSU__DDRC__DQMAP_68_71                   {0}             \
    CONFIG.PSU_DDR_RAM_HIGHADDR                     {0xFFFFFFFF}    \
    CONFIG.PSU_DDR_RAM_HIGHADDR_OFFSET              {0x800000000}   \
    CONFIG.PSU_DDR_RAM_LOWADDR_OFFSET               {0x80000000}    \
    CONFIG.PSU__UART0__PERIPHERAL__ENABLE           {1}             \
    CONFIG.PSU__UART0__PERIPHERAL__IO               {MIO 18 .. 19}  \
    CONFIG.PSU__UART1__PERIPHERAL__ENABLE           {1}             \
    CONFIG.PSU__UART1__PERIPHERAL__IO               {MIO 20 .. 21}  \
    CONFIG.PSU__USB0__PERIPHERAL__ENABLE            {1}             \
    CONFIG.PSU__USB0__REF_CLK_SEL                   {Ref Clk2}      \
    CONFIG.PSU__DP__REF_CLK_SEL                     {Ref Clk3}      \
    CONFIG.PSU__USB3_0__PERIPHERAL__ENABLE          {1}             \
    CONFIG.PSU__USB3_0__PERIPHERAL__IO              {GT Lane2}      \
    CONFIG.PSU_BANK_0_IO_STANDARD                   {LVCMOS18}      \
    CONFIG.PSU_BANK_1_IO_STANDARD                   {LVCMOS18}      \
    CONFIG.PSU_BANK_2_IO_STANDARD                   {LVCMOS18}      \
    CONFIG.PSU_BANK_3_IO_STANDARD                   {LVCMOS18}      \
    CONFIG.PSU__DISPLAYPORT__PERIPHERAL__ENABLE     {1}             \
    CONFIG.PSU__SATA__PERIPHERAL__ENABLE            {1}             \
    CONFIG.PSU__SATA__LANE1__IO                     {GT Lane3}      \
    CONFIG.PSU__SATA__REF_CLK_FREQ                  {125}           \
                                                                    \
    CONFIG.PSU__PSS_REF_CLK__FREQMHZ                {33.330}        \
    CONFIG.PSU__CRF_APB__DDR_CTRL__FREQMHZ          {1067}          \
    CONFIG.PSU__CRF_APB__DP_VIDEO_REF_CTRL__SRCSEL  {VPLL}          \
    CONFIG.PSU__CRF_APB__DP_AUDIO_REF_CTRL__SRCSEL  {RPLL}          \
    CONFIG.PSU__CRF_APB__DP_STC_REF_CTRL__SRCSEL    {RPLL}          \
    CONFIG.PSU__CRF_APB__GPU_REF_CTRL__SRCSEL       {IOPLL}         \
    CONFIG.PSU__CRF_APB__GDMA_REF_CTRL__SRCSEL      {APLL}          \
    CONFIG.PSU__CRF_APB__DPDMA_REF_CTRL__SRCSEL     {APLL}          \
    CONFIG.PSU__CRF_APB__TOPSW_MAIN_CTRL__SRCSEL    {DPLL}          \
    CONFIG.PSU__CRL_APB__SDIO1_REF_CTRL__SRCSEL     {IOPLL}         \
    CONFIG.PSU__CRL_APB__IOU_SWITCH_CTRL__SRCSEL    {IOPLL}         \
    CONFIG.PSU__CRL_APB__LPD_SWITCH_CTRL__SRCSEL    {IOPLL}         \
    CONFIG.PSU__CRL_APB__ADMA_REF_CTRL__SRCSEL      {IOPLL}         \
    CONFIG.PSU__CRL_APB__TIMESTAMP_REF_CTRL__SRCSEL {IOPLL}         \
    CONFIG.PSU__CRL_APB__CPU_R5_CTRL__SRCSEL        {IOPLL}         \
    CONFIG.PSU__CRF_APB__ACPU_CTRL__FREQMHZ         {1200}          \
    CONFIG.PSU__CRF_APB__GPU_REF_CTRL__FREQMHZ      {500}           \
    CONFIG.PSU__CRL_APB__QSPI_REF_CTRL__FREQMHZ     {125}           \
    CONFIG.PSU__CRL_APB__CPU_R5_CTRL__FREQMHZ       {500}           \
    CONFIG.PSU__CRL_APB__IOU_SWITCH_CTRL__FREQMHZ   {250}           \
    CONFIG.PSU__CRL_APB__LPD_SWITCH_CTRL__FREQMHZ   {500}           \
    CONFIG.PSU__CRL_APB__ADMA_REF_CTRL__FREQMHZ     {500}           \
                                                                    \
    CONFIG.PSU__USE__IRQ0                           {1}             \
    CONFIG.PSU__USE__IRQ1                           {1}             \
        ]	  

      lib_puts BLK "Loaded base tcl [lib_rela_path $base_path]"
    }
    # need check prop
    set chkp_need     [dict exists $ip_info CHKP]
    if {$chkp_need == 1 } {set chkp_keys [dict keys   [dict get $ip_info CHKP]] ;\
                           set chkp_vals [dict values [dict get $ip_info CHKP]] 
    } else                {set chkp_keys {} ;set chkp_vals {}          
    }
    # print table title
    set col_list      [list PROPERTY VALUE]
    set width_list    [list [lib_llength_max [concat [dict keys   [dict get $ip_info PROP] ] [lindex $col_list 0] $chkp_keys  ] ] \
                            [lib_llength_max [concat [dict values [dict get $ip_info PROP] ] [lindex $col_list 1] $chkp_vals  ] ] ]
    set fmtstr_cont   [lib_tbl_frmstr_content $width_list]
    set fmtstr_boder  [lib_tbl_frmstr_border  $width_list]
    lib_tbl_title_bar BLK $fmtstr_boder $fmtstr_cont $col_list
    # print table prop
    dict for { prop value } [dict get $ip_info PROP] {
      dict set dict_prop "CONFIG.$prop" $value
      lib_puts BLK [format $fmtstr_cont $prop $value]
    }
    if {[llength [dict keys $dict_prop]]>0} {
      set_property  -dict $dict_prop [get_bd_cells $path]
    }
    lib_tbl_border BLK $fmtstr_boder
    # print table check prop
    if {$chkp_need == 1 } {
      dict for { prop value } [dict get $ip_info CHKP] {
        lib_puts BLK [format $fmtstr_cont $prop [get_property  "CONFIG.$prop" [get_bd_cells $path]] ]
      }
      lib_tbl_border BLK $fmtstr_boder    
    }
    lib_puts BLK "Successfully Created ip $path!"
  }
  lib_puts BLK "Successfully Created all ips!"
}

#****************************************************************
# add files
#****************************************************************
proc lib_add_files { file_set files } {
  if {[llength [dict keys $files]] < 1 } {
    lib_puts ADD "Adding None files"
  } else {
    lib_puts ADD "Adding files..."
    set col_list    $file_set
    set width_file   [lindex $col_list 0]
    dict for { file_name file_info } $files {
      dict with file_info {
        add_files -fileset $file_set -norecurse $PATH
        lappend width_file [lib_rela_path $PATH]
      }
    }
    set width_list    [lib_llength_max $width_file]
    set fmtstr_cont   [lib_tbl_frmstr_content $width_list]
    set fmtstr_boder  [lib_tbl_frmstr_border  $width_list]
    lib_tbl_title_bar ADD $fmtstr_boder $fmtstr_cont $col_list
    foreach file [lrange $width_file 1 end] {
      lib_puts ADD [format $fmtstr_cont $file]
    }
    lib_tbl_border ADD $fmtstr_boder
    lib_puts ADD "Successfully added all files into $file_set fileset!"
  }
}

#****************************************************************
# format string content
#****************************************************************
proc lib_tbl_frmstr_content { len_list } {
  set frmstr "|"
  foreach content $len_list {
    append frmstr " %-$content\s |"
  }
  return $frmstr
}

#****************************************************************
# format string border
#****************************************************************
proc lib_tbl_frmstr_border { len_list } {
  set frmstr "+"
  foreach content $len_list {
    append frmstr "[string repeat - [expr $content+2]]+"
  }
  return $frmstr
}

#****************************************************************
# table border
#****************************************************************
proc lib_tbl_border {label fmtstr_boder {chan stdout}} {
  lib_puts $label [format $fmtstr_boder] $chan
}

#****************************************************************
# table title bar
#****************************************************************
proc lib_tbl_title_bar {label fmtstr_boder fmtstr_cont col_list {chan stdout} } {
  lib_tbl_border  $label $fmtstr_boder $chan
  lib_puts        $label [format $fmtstr_cont {*}$col_list] $chan
  lib_tbl_border  $label $fmtstr_boder $chan
}

#****************************************************************
# puts
#****************************************************************
proc lib_puts { label content {chan stdout} } {
  puts $chan "\[$label\] $content"
}

#****************************************************************
# puts nonewline
#****************************************************************
proc lib_puts_nonewline { label content} {
  puts -nonewline "\[$label\] $content"
}

#****************************************************************
# puts error
#****************************************************************
proc lib_error { label content} {
  error "\[$label-ERROR\] $content"
}

#****************************************************************
# current time
#****************************************************************
proc lib_time { fmt} {
  return [clock format [clock seconds] -format $fmt]
}

#****************************************************************
# string relative path
#****************************************************************
proc lib_rela_path { path } {
  set nml_path [file normalize $path]
  set map_from [lib_sys pwd_dir]
  set map_to   .
  set map_path $nml_path
  set map_time 0
  while {$nml_path == $map_path} {
    if {$map_time>0} {
      set map_from [file dirname $map_from]
      set map_to   "$map_to/.."
    }
    set map_path [string map [list $map_from $map_to] $nml_path]
    incr map_time
  }
  return  $map_path
}

#****************************************************************
# list max bytelength
#****************************************************************
proc lib_llength_max {list_in} {
  set max_length 0
  foreach ele $list_in {
    set len [string bytelength $ele]
    if {$len>$max_length} {set max_length $len}
  }
  return $max_length
}

#****************************************************************
# create project
#****************************************************************
proc lib_create_prj {prj_name prj_dir prj_part} {
  lib_puts PRJ "Creating project $prj_name..."
  create_project $prj_name -dir $prj_dir -part $prj_part -force
  lib_puts PRJ "Successfully created project $prj_name!"
}

#****************************************************************
# open project
#****************************************************************
proc lib_open_prj {prj_name prj_dir} {
  lib_puts PRJ "Opening project $prj_name..."
  open_project $prj_dir/$prj_name -quiet
  lib_puts PRJ "Successfully opened project $prj_name!"
}

#****************************************************************
# check exist
#****************************************************************
proc lib_chk_exist { type file_path op_open op_recreate op_create} {
  if {[file exists $file_path] == 1 } {
    set file_name [file tail $file_path]
    lib_puts PRJ "The $type $file_name already exists!"
    lib_puts_nonewline PRJ "Enter \"O\" to open it, enter \"R\" to recreate: "
    set keyin [gets stdin]
    switch -exact -- $keyin {
      {O} - {o} {uplevel $op_open                           }
      {R} - {r} {uplevel $op_recreate ; uplevel $op_create  }
      default   {lib_puts ERR "error input";return -code error -level 10}
    }
  } else {
    uplevel $op_create
  }
}
#****************************************************************
# create pin
#****************************************************************
proc lib_create_pin {pin_name pin_dir pin_type pin_vector pin_from pin_to prop} {
  if ($pin_vector==1) {create_bd_pin -dir $pin_dir -type $pin_type -from $pin_from -to $pin_to $pin_name 
  } else              {create_bd_pin -dir $pin_dir -type $pin_type                             $pin_name
  }
  if { $prop != {} } {
    dict for { prop_name prop_value } $prop {
      set_property "CONFIG.$prop_name" $prop_value [get_bd_pins $pin_name]
    }
  } 
}

#****************************************************************
# create port
#****************************************************************
proc lib_create_port {port_name port_dir port_type port_vector port_from port_to prop} {
  if {$port_vector==1}  {create_bd_port -dir $port_dir -type $port_type -from $port_from -to $port_to $port_name 
  } else                {create_bd_port -dir $port_dir -type $port_type                               $port_name 
  }
  if { $prop != {} } {
    lib_puts PIN "Setting properties of $port_name"
    dict for { prop_name prop_value } $prop {
      set_property "CONFIG.$prop_name" $prop_value [get_bd_ports $port_name]
      lib_puts PIN "Set $prop_name to $prop_value"
    }
    lib_puts PIN "Successfully set all properties of $port_name"
  } 
}

#****************************************************************
# create_pins
#****************************************************************
proc lib_create_pins { dict_pin } {
  lib_puts  PIN "Creating pins..."
  set col_list      [list CLASS PATH]
  set width_class   [lindex $col_list 0]
  set width_path    [lindex $col_list 1]
  dict for { pin_name pin_info } $dict_pin {
    dict with pin_info {
      if { [info exists PROP] != {1} } { set PROP {} }
      switch -exact -- $CLASS {
        PIN       {lib_create_pin       $PATH $DIR  $TYPE $VECTOR $FROM $TO $PROP }
        PORT      {lib_create_port      $PATH $DIR  $TYPE $VECTOR $FROM $TO $PROP }
        INTF_PIN  {create_bd_intf_pin   -mode $MODE -vlnv $VLNV $PATH         }
        INTF_PORT {create_bd_intf_port  -mode $MODE -vlnv $VLNV $PATH         }
        default   {lib_puts ERR         "Incorrect PIN CLASS..."              }   
      }
      lappend width_class $CLASS
      lappend width_path  $PATH
      unset PROP
    }
  }
  set width_list    [list [lib_llength_max $width_class] [lib_llength_max $width_path] ]
  set fmtstr_cont   [lib_tbl_frmstr_content $width_list]
  set fmtstr_boder  [lib_tbl_frmstr_border  $width_list]
  lib_tbl_title_bar PIN $fmtstr_boder $fmtstr_cont $col_list
  dict for { pin_name pin_info } $dict_pin {
    dict with pin_info {
      lib_puts PIN [format $fmtstr_cont $CLASS $PATH]
    }
  }
  lib_tbl_border PIN $fmtstr_boder
  lib_puts PIN "Successfully Created all pins!"
}

#****************************************************************
# connect interface net
#****************************************************************
proc lib_connect_intf_net {src_type src_intf dst_type dst_intf} {
  if        {$src_type=={PIN} } {set src [get_bd_intf_pins  $src_intf]
  } elseif  {$src_type=={PORT}} {set src [get_bd_intf_ports $src_intf]
  } else    {lib_error ERR "Unsupported SRC_TYPE $src_type of SRC_INTF $src_intf" 
  }
  if        {$dst_type=={PIN} } {set dst [get_bd_intf_pins  $dst_intf]
  } elseif  {$dst_type=={PORT}} {set dst [get_bd_intf_ports $dst_intf]
  } else    {lib_error ERR "Unsupported DST_TYPE $dst_type of DST_INTF $dst_intf" 
  }
  if { $src != {} && $dst != {} } {
    connect_bd_intf_net $src $dst
  } else {
    lib_puts WRN "Connection from $src_pin to $dst_pin is invalid..."
  }
}

#****************************************************************
# connect net
#****************************************************************
proc lib_connect_net {src_type src_pin dst_type dst_pin} {
  if        {$src_type=={PIN} } {set src [get_bd_pins  $src_pin]
  } elseif  {$src_type=={PORT}} {set src [get_bd_ports $src_pin]
  } else    {lib_error ERR "Unsupported SRC_TYPE $src_type of SRC_PIN $src_pin" 
  }
  if        {$dst_type=={PIN} } {set dst [get_bd_pins  $dst_pin]
  } elseif  {$dst_type=={PORT}} {set dst [get_bd_ports $dst_pin]
  } else    {lib_error ERR "Unsupported DST_TYPE $dst_type of DST_PIN $dst_pin" 
  }
  if { $src != {} && $dst != {} } {
    connect_bd_net $src $dst
  } else {
    lib_puts WRN "Connection from $src_pin to $dst_pin is invalid..."
  }
}

#****************************************************************
# connect nets
#****************************************************************
proc lib_connect { dict_cn } {
  dict for { cn_mod cn_info } $dict_cn {
    lib_puts  PIN "Connecting nets for $cn_mod..."
    set dst_list    "DST_INTF_PORT"
    set src_list    "SRC_INTF_PORT"
    dict for { type conn } $cn_info {
      set dst_list  [concat $dst_list [dict keys   $conn]]
      set src_list  [concat $src_list [dict values $conn]]
    }
    set width_list    [list [lib_llength_max $dst_list  ] \
                            [lib_llength_max $src_list  ] ]
    dict for { type conn } $cn_info {
      scan $type "%s %s %s" dst_type net_type src_type
      if {$net_type=={intf}}  {set col_list [list "DST_INTF_$dst_type"  "SRC_INTF_$src_type"  ]
      } else                  {set col_list [list "DST_$dst_type"       "SRC_$src_type"       ]
      }
      set fmtstr_cont   [lib_tbl_frmstr_content $width_list]
      set fmtstr_boder  [lib_tbl_frmstr_border  $width_list]
      lib_tbl_title_bar PIN $fmtstr_boder $fmtstr_cont $col_list
      dict for { dst src } $conn {
        switch -exact -- $net_type {
          from  {lib_connect_net      $src_type $src $dst_type $dst}
          intf  {lib_connect_intf_net $src_type $src $dst_type $dst}
        }
        lib_puts PIN [format $fmtstr_cont $dst $src ]
      }
      lib_tbl_border PIN $fmtstr_boder
    }
    lib_puts PIN "Successfully connected all nets for $cn_mod!"
  }
  lib_puts PIN "Successfully connected all nets for the project!"
}

#****************************************************************
# set reg addresses
#****************************************************************
proc lib_map_reg_addrs {addrs} {
  lib_puts PRJ "Mapping axi REG address..."
  dict for {addr_type addr_info} $addrs {
    if { [dict exists $addr_info REG] == {1} } {
      dict with addr_info {
        set all_regs  [get_bd_addr_segs]
        assign_bd_address [get_bd_addr_segs -hier $REG] -range $RANGE -offset $OFFSET -quiet
        set all_regs_add [get_bd_addr_segs]
        if { [llength $all_regs] != [llength $all_regs_add] } {
          foreach old_reg $all_regs {
            set ind_old [lsearch -exact $all_regs_add $old_reg]
            if { $ind_old >= {0} } {
              set all_regs_add [lreplace $all_regs_add $ind_old $ind_old]
            }
          }
        }
        set reg_name  [lindex $all_regs_add 0]
        set_property offset $OFFSET         [get_bd_addr_segs -hier $reg_name] 
        set_property range  $RANGE          [get_bd_addr_segs -hier $reg_name] 
        set_property offset $OFFSET         [get_bd_addr_segs -hier $reg_name] 
        lib_puts PRJ "Mapped axi reg address $REG at < $OFFSET [ $RANGE ] >"
      }
    } elseif { [dict exists $addr_info HP] == {1}} {
      dict with addr_info {
        foreach base $BASENAME {
          set hp_base [get_bd_addr_segs -hier $HP/$base]
          assign_bd_address $hp_base 
          lib_puts PRJ "Mapped axi reg address $hp_base"
        }
      }
    }  
  }
  lib_puts PRJ "Successfully Mapped all axi REG addresses!"
  lib_puts PRJ "Mapping axi HP address..."
  foreach addr [get_bd_addr_segs] {
    assign_bd_address $addr -quiet
  }
  set excluded  [get_bd_addr_segs -excluded]
  foreach ex $excluded {
    if {[string match *SEG*LPS_OCM* $ex]} {
      include_bd_addr_seg $ex -quiet
    }
  }
  lib_puts PRJ "Successfully Mapped all axi HP addresses automatically!"
  
}

#****************************************************************
# save_bd_design
#****************************************************************
proc lib_save_bd {} {
  global dict_prj
  dict with dict_prj {
    if {[info exists dict_hier]} {
      dict for { hier_key hier_info } $dict_hier {
        regenerate_bd_layout -hierarchy [lib_hier $hier_key]
      }
    }
  }
  regenerate_bd_layout
  save_bd_design
}

#****************************************************************
# make wrapper
#****************************************************************
proc lib_make_wrapper {bd_path bd_wrapper_path} {
  make_wrapper -files [get_files $bd_path] -top
  add_files -norecurse $bd_wrapper_path
  lib_puts PRJ "Successfully made at [lib_rela_path $bd_wrapper_path]!"
}

#****************************************************************
# create ip run
#****************************************************************
proc lib_create_run_ip {bd_path source_set} {
  create_ip_run [get_files -of_objects [get_fileset $source_set] $bd_path]
  lib_puts RUN "Successfully created ip run for [lib_rela_path $bd_path]!"
}

#****************************************************************
# create ip strategy
#****************************************************************
proc lib_create_ip_stgy {bd_name jobs} {
  global dict_prj
  if {([dict exists $dict_prj dict_stgy synth_ip_stgy STRATEGY] != {1})||([dict get $dict_prj dict_stgy synth_ip_stgy STRATEGY] == {}) } {
    dict set dict_prj dict_stgy synth_ip_stgy STRATEGY {Vivado Synthesis Defaults}
  } 
  set stgy [dict get $dict_prj dict_stgy synth_ip_stgy STRATEGY]
  set filter "\{NAME=~*$bd_name*synth*\}"
  foreach run [get_runs -filter [expr $filter]] {
    dict set dict_prj dict_stgy synth_ip $run [dict create "STRATEGY" $stgy "JOBS" $jobs]
    set_property STRATEGY $stgy [get_runs $run]
  }
  lib_puts RUN "Successfully created ip run list for $bd_name!"
  lib_puts RUN "Successfully set all ip run strategies as $stgy!"
}

#****************************************************************
# create ip custom strategy 
#****************************************************************
proc lib_create_ip_custom {} {
  global dict_prj
  dict for { cell ip_info } [dict get $dict_prj dict_stgy synth_ip_custom] {
    set cell_name [string range $cell [expr [string last "/" $cell]+1] [string length $cell]]
    set filter    "\{NAME=~*$cell_name*synth*\}"
    foreach run [get_runs -filter [expr $filter]] {
      dict unset  dict_prj dict_stgy synth_ip_custom $cell
      dict set    dict_prj dict_stgy synth_ip_custom $run $ip_info
    }
  }
  lib_puts RUN "Successfully set ip custom strategies list!"
}

#****************************************************************
# run synths
#****************************************************************
proc lib_run_synths { synths } {
  if {[dict size $synths]==0} {
    lib_puts RUN "Launching None synth run"
  } else {
    lib_puts RUN "Launching synth runs..."
    set col_list      [list NAME STRATEGY JOBS]
    set width_list    [lib_tbl_width_list_value $col_list $synths]
    set fmtstr_cont   [lib_tbl_frmstr_content $width_list]
    set fmtstr_boder  [lib_tbl_frmstr_border  $width_list]
    lib_tbl_title_bar RUN $fmtstr_boder $fmtstr_cont $col_list
    dict for { run_name run_info } $synths {
      dict with run_info {
        lib_run_synth $run_name $JOBS
        lib_puts RUN [format $fmtstr_cont $run_name $STRATEGY $JOBS]
      }
    }
    lib_tbl_border RUN $fmtstr_boder
    lib_puts RUN "Successfully launched all synth runs!"
  }
}

#****************************************************************
# wait on run
#****************************************************************
package require control 
namespace import control::do 

proc lib_wait_runs { stgy type } {
  global dict_prj
  ## generate runlist
  set run_list ""
  if {$type=={synth}} {
    set run_list [dict keys $stgy]
  } elseif {$type=={impl}} {
    set run_list [concat [lsort [lib_dict_value_picked $stgy {} PARENT]] [dict keys $stgy] ]
  } else {
    lib_puts RUN "Incorrect agg run type $type"
    return ""
  }
  ## generate table format string 
  set col_list        [list NAME 100%% STATUS]
  set len_run         [lib_llength_max $run_list]
  set len_progress    3
  set len_status      50
  set width_list      [list $len_run [expr $len_progress+1] $len_status]
  set fmtstr_cont     "| %-$len_run\s | %$len_progress\.0f%% | %-$len_status\s |"
  set fmtstr_title    [lib_tbl_frmstr_content $width_list]
  set fmtstr_boder    [lib_tbl_frmstr_border  $width_list]
  set run_total       [llength $run_list]
  ## generate table
  set run_first_time  1
  set progress_pre    {}
  set status_pre      {}
  set s_queue_pre     {}
  set cnt_nofreash      0
  set cnt_nofreash_max  60
  do {
    ## get data
    set progress  [get_property PROGRESS  [get_runs $run_list]]
    set status    [get_property STATUS    [get_runs $run_list]]
    set p_finish  [lsearch      -all $progress "100%"]
    set p_running [lsearch -not -all $progress "100%"]
    set s_running {}
    set s_error   {}
    set s_idle    {}
    set p_idle    {}
    set s_queue   {}
    set s_cancel  {}
    foreach p $p_running {
      set s [lindex $status $p]
            if {[string match *ERROR*       $s]}  {lappend s_error   $s
      } elseif {[string match *Script*      $s]}  {lappend s_idle    $s ; lappend p_idle  $p
      } elseif {[string match *Queued*      $s]}  {lappend s_queue   $s
      } elseif {[string match *Not\ start*  $s]}  {lappend s_cancel  $s
      } else                                      {lappend s_running $s
      } 
    }
    if { ($progress != $progress_pre) || (($status != $status_pre)) } {
      if { ($type=={impl}) && ($s_queue != $s_queue_pre) && ([llength $p_idle] > 0) } {
        set impl_queue_base [expr [llength $run_list] - [llength [dict keys $stgy]] ]
        foreach p_idle_ind $p_idle {
          if { $p_idle_ind >= $impl_queue_base } {
            set run_name [lindex $run_list $p_idle_ind]
            dict with stgy $run_name {
              lib_run_impl $run_name $PARENT $JOBS
            }
          }
        }
      }
      set s_queue_pre $s_queue

      set cnt_nofreash 0
      ## push back
      if {$run_first_time==0} {
        lib_tbl_line_back [expr $run_total+2+3]
      } else  { 
        set run_first_time 0
        lib_puts RUN "Starting run at \[[lib_time "%Y%m%d-%H:%M:%S"]\]"
      }
      lib_tbl_title_bar RUN $fmtstr_boder $fmtstr_title $col_list
      ## puts table
      foreach run $run_list {
        set prog [get_property PROGRESS [get_runs $run]]
        scan $prog "%f" prog_i
        set status_run "[get_property STATUS [get_runs $run]] [get_property {STATS.WNS} [get_runs $run]]"
        lib_puts RUN [format $fmtstr_cont $run $prog_i $status_run]
      }
      ## puts border
      lib_tbl_border RUN $fmtstr_boder
      lib_puts RUN "Checking run at \[[lib_time "%Y%m%d-%H:%M:%S"]\]"
    } else {
      if { $cnt_nofreash < $cnt_nofreash_max } {
        incr cnt_nofreash 
      } else {
        lib_tbl_line_back 1
        lib_puts RUN "Checking run at \[[lib_time "%Y%m%d-%H:%M:%S"]\]"
        set cnt_nofreash 0
      }
    }
    set progress_pre  $progress
    set status_pre    $status
    lib_sleep_s 1 
  } while { ([llength $s_running]>0) || ([llength $p_finish]==0) }
  ## if synth is not finished, the while loop will not exit
  if {[llength $s_error] == 0} {
    lib_puts RUN "Successfully finished all runs!"
    dict set dict_prj dict_sys run_error 0
  } else {
    lib_puts RUN "Found ERROR!"
    dict set dict_prj dict_sys run_error 1
  }
}


#****************************************************************
# column width list: get each key's value
#****************************************************************
proc lib_tbl_width_list_value { col_list content_dict} {
  set width_list [lib_llength_max [concat [lindex $col_list 0] [dict keys $content_dict] ]]
  foreach col [lrange $col_list 1 end] {
    set l {}
    dict for {name name_info} $content_dict {
      lappend l [dict get $name_info $col]
    }
    lappend l $col
    lappend width_list [lib_llength_max $l]
  }
  return $width_list
}

#****************************************************************
# run synth
#****************************************************************
proc lib_run_synth {synth_name synth_jobs} {
  # lib_puts RUN "Launching synth run $synth_name at $synth_jobs jobs..."
  launch_runs $synth_name -jobs $synth_jobs -quiet
  # lib_tbl_line_back 1
}

#****************************************************************
# sleep seconds
#****************************************************************
proc lib_sleep_s {N} {
  after [expr {int($N * 1000)}]
}

#****************************************************************
# line back
#****************************************************************
proc lib_tbl_line_back {total} {
  set line_back_base  [string repeat "\r\b" $total]
  set line_back       [append line_back_base "\r"]
  puts -nonewline $line_back
}

#****************************************************************
# set run strategy
#****************************************************************
proc lib_set_stgy { step } {
  global dict_prj
  dict for {stgy_name stgy_info } [dict get $dict_prj dict_stgy $step] {
    set num [lib_get_stgy_num $step $stgy_name]
    switch -exact -- $step {
      synth {
        if {([dict exists $stgy_info STRATEGY] != {1})||([dict get $stgy_info STRATEGY] == {}) } {
          dict set dict_prj dict_stgy $step $stgy_name STRATEGY [lib_get_stgy_synth [lindex $num 0]]
        }
      }
      impl {
        if {([dict exists $stgy_info PARENT] != {1})||([dict get $stgy_info PARENT] == {}) } {
          dict set dict_prj dict_stgy $step $stgy_name PARENT "synth_[lindex $num 0]"
          dict set dict_prj dict_stgy $step $stgy_name PARENT_STGY [lib_get_stgy_synth [lindex $num 0]]
        } elseif {([dict exists $stgy_info PARENT_STGY] != {1})||([dict get $stgy_info PARENT_STGY] == {}) } {
          set parent [dict get $stgy_info PARENT]
          set parent_stgy [dict get $dict_prj dict_stgy synth $parent STRATEGY]
          dict set dict_prj dict_stgy $step $stgy_name PARENT_STGY $parent_stgy
        }
        if {([dict exists $stgy_info STRATEGY] != {1})||([dict get $stgy_info STRATEGY] == {}) } {
          dict set dict_prj dict_stgy $step $stgy_name STRATEGY [lib_get_stgy_impl [lindex $num 1]]
        }  
      }
      default {lib_puts ERR "error step $step!";return -code error -level 10}
    } 
  }  
}

#****************************************************************
# create run synths
#****************************************************************
proc lib_create_run_synths {synths } {
  if {[dict size $synths]==0} {
    lib_puts RUN "Creating None synth run"
  } else {
    set flow  "Vivado Synthesis [string range [version -short] 0 3]"
    lib_puts RUN "Creating synth runs..."
    set col_list      [list RUN STRATEGY]
    set width_list    [lib_tbl_width_list_value $col_list $synths]
    set fmtstr_cont   [lib_tbl_frmstr_content $width_list]
    set fmtstr_boder  [lib_tbl_frmstr_border  $width_list]
    lib_tbl_title_bar RUN $fmtstr_boder $fmtstr_cont $col_list
    dict for {run_name run_info} $synths {
      dict with run_info {
        if { $run_name == {synth_1} } {
          set_property strategy $STRATEGY [get_runs $run_name]
        } else {
          create_run $run_name -flow $flow -strategy $STRATEGY -quiet
        }
        lib_puts RUN [format $fmtstr_cont $run_name $STRATEGY]
      }
    }
    lib_tbl_border RUN $fmtstr_boder
    lib_puts RUN "Successfully created all synth runs!"
  }
}

#****************************************************************
# create run impls
#****************************************************************
proc lib_create_run_impls {impls } {
  if {[dict size $impls]==0} {
    lib_puts RUN "Creating None impl run"
  } else {
    set flow  "Vivado Implementation [string range [version -short] 0 3]"
    lib_puts RUN "Creating impl runs..."
    set col_list      [list RUN PARENT_STGY STRATEGY]
    set width_list    [lib_tbl_width_list_value $col_list $impls]
    set fmtstr_cont   [lib_tbl_frmstr_content $width_list]
    set fmtstr_boder  [lib_tbl_frmstr_border  $width_list]
    lib_tbl_title_bar RUN $fmtstr_boder $fmtstr_cont $col_list
    dict for {run_name run_info} $impls {
      dict with run_info {
        create_run $run_name -flow $flow -strategy $STRATEGY -parent_run $PARENT -quiet
        lib_puts RUN [format $fmtstr_cont $run_name $PARENT_STGY $STRATEGY]
      }
    }
    lib_tbl_border RUN $fmtstr_boder
    lib_puts RUN "Successfully created all impl runs!"
  }
}

#****************************************************************
# current run
#****************************************************************
proc lib_current_run { runs step } {
  set run_name [lindex [dict keys $runs] 0]
  current_run "-$step"  [get_run $run_name] -quiet
  lib_puts RUN "Successfully set current $step run as $run_name!"
  delete_runs impl_1 -quiet
}

#****************************************************************
# get strategy num
#****************************************************************
proc lib_get_stgy_num { step run_name } {
  set num [split $run_name _]
  set num [lrange $num 1 end]
  return $num
}

#****************************************************************
# get strategy synth
#****************************************************************
proc lib_get_stgy_synth { num } {
  set ver [version -short]
  set stgy {}
  switch -exact -- $ver {
    {2017.3}  -
    default   {
      switch -exact -- $num {
        1 {set stgy {Vivado Synthesis Defaults} }
        2 {set stgy {Flow_AreaOptimized_high}   }
        3 {set stgy {Flow_AreaOptimized_medium} }
        4 {set stgy {Flow_AreaMultThresholdDSP} }
        5 {set stgy {Flow_AlternateRoutability} }
        6 {set stgy {Flow_PerfOptimized_high}   }
        7 {set stgy {Flow_PerfThresholdCarry}   }
        8 {set stgy {Flow_RuntimeOptimized}     }
        default {lib_puts ERR "error synth strategy num $num!";return -code error -level 10}
      }
    }
  }
  return $stgy
}

#****************************************************************
# get strategy impl
#****************************************************************
proc lib_get_stgy_impl { num } {
  set ver [version -short]
  set stgy {}
  switch -exact -- $ver {
    {2017.3}  -
    default   {
      switch -exact -- $num {
        01 {set stgy {Vivado Implementation Defaults}         }
        02 {set stgy {Performance_Explore}                    }
        03 {set stgy {Performance_ExplorePostRoutePhysOpt}    }
        04 {set stgy {Performance_WLBlockPlacement}           }
        05 {set stgy {Performance_WLBlockPlacementFanoutOpt}  }
        06 {set stgy {Performance_EarlyBlockPlacement}        }
        07 {set stgy {Performance_NetDelay_high}              }
        08 {set stgy {Performance_NetDelay_low}               }
        09 {set stgy {Performance_Retiming}                   }
        10 {set stgy {Performance_ExtraTimingOpt}             }
        11 {set stgy {Performance_RefinePlacement}            }
        12 {set stgy {Performance_SpreadSLLs}                 }
        13 {set stgy {Performance_BalanceSLLs}                }
        14 {set stgy {Congestion_SpreadLogic_high}            }
        15 {set stgy {Congestion_SpreadLogic_medium}          }
        16 {set stgy {Congestion_SpreadLogic_low}             }
        17 {set stgy {Congestion_SpreadLogic_Explore}         }
        18 {set stgy {Congestion_SSI_SpreadLogic_high}        }
        19 {set stgy {Congestion_SSI_SpreadLogic_low}         }
        20 {set stgy {Congestion_SSI_SpreadLogic_Explore}     }
        21 {set stgy {Area_Explore}                           }
        22 {set stgy {Area_ExploreSequential}                 }
        23 {set stgy {Area_ExploreWithRemap}                  }
        24 {set stgy {Power_DefaultOpt}                       }
        25 {set stgy {Power_ExploreArea}                      }
        26 {set stgy {Flow_RunPhysOpt}                        }
        27 {set stgy {Flow_RunPostRoutePhysOpt}               }
        28 {set stgy {Flow_RuntimeOptimized}                  }
        29 {set stgy {Flow_Quick}                             }
        default {lib_puts ERR "error impl strategy num $num!";return -code error -level 10}
      }
    }
  }
  return $stgy
}

#****************************************************************
# load ip repo
#****************************************************************
proc lib_load_iprepo { } {
  global dict_prj

  set ip_dir [lib_sys ip_dir]

  lib_puts BLK "Loading ip repository..."
  set_property  ip_repo_paths  $ip_dir [current_project]
  update_ip_catalog -quiet
  lib_puts BLK "Successfully loaded ip repositories in [lib_rela_path $ip_dir]!"
}

#****************************************************************
# build bd
#****************************************************************
proc lib_build_bd { } {
  global dict_prj
  
  set op_open {
    lib_open_prj   $prj_name $prj_dir
    lib_open_bd    $bd_name  $bd_path
    return -code 0 -level 2
  }  
  set op_create {
    lib_create_prj $prj_name $prj_dir  $prj_part
  } 
  set op_delete {
    file delete -force $prj_dir
    lib_puts PRJ "The old project has been deleted!"
  } 
  set op_move {
    file rename $prj_temp_dir $new_path 
    lib_puts PRJ "The old project has been moved to $new_path_sht!"
  } 
  set op_recreate {
    file mkdir $backup_dir 
    lib_puts_nonewline PRJ "Enter annotation of the old project, or \"ESC\" to skip backup: " 
    set keyin [gets stdin] 
    set backup_flag 1
    if {[llength $keyin]>{0}} {
      if {$keyin == "\x1b"} { set backup_flag 0 
      } else {                set new_path $new_path\_$keyin  
                              set new_path_sht  [lib_rela_path $new_path]
      }
    } else {                  lib_puts PRJ "The old project has no annotation. Please rename it later!"
    }
    if {$backup_flag=={0}}  { uplevel #2 $op_delete
    } else {                  uplevel #2 $op_move
    }
  }

  dict with dict_prj dict_sys {
    uplevel #2 $op_delete
    uplevel #2 $op_create

    lib_create_bd     $bd_name $bd_dir $bd_ooc $bd_path
	if { [lib_sys flow_edt_bd] == 1 || [lib_sys flow_crt_bd] ==1 } {
      file copy -force $work_dir $prj_temp_dir/work
	}
    if { [dict exists $dict_prj dict_src_mod] == {1} } {
      dict for { src prop } [dict get $dict_prj dict_src_mod] {
        dict with prop {
          exec sed -i "s/$REPLACE/$VALUE/" $FILE
          lib_puts BLK "Successfully replaced in [lib_rela_path $FILE] ."
        }
      }
    }
    if { [lib_sys flow_edt_bd] == 1 || [lib_sys flow_crt_bd] ==1 } {
      if { [dict exists $dict_prj dict_ip ps] == {1} } {
        file mkdir $prj_temp_dir/base
        file copy -force [lib_rela_path [lib_value ps BASE]] $prj_temp_dir/base
      }
    }

    lib_load_iprepo
  }
  dict with dict_prj {
    if {[info exists dict_src ]} {  lib_add_files     {sources_1} $dict_src }
    if {[info exists dict_xdc ]} {  lib_add_files     {constrs_1} $dict_xdc }
    if {[info exists dict_hier]} {  lib_create_hiers  $dict_hier  }
    if {[info exists dict_ip  ]} {  lib_create_ips    $dict_ip    }
    if {[info exists dict_pin ]} {  lib_create_pins   $dict_pin   }
    if {[info exists dict_cn  ]} {  lib_connect       $dict_cn    }
    if {[info exists dict_addr]} {  lib_map_reg_addrs $dict_addr  }
  }
  lib_save_bd
  lib_make_wrapper        [lib_sys bd_path] [lib_sys bd_wrapper_path]

  if { [dict exists $dict_prj dict_ip] == {1} } {
    lib_puts BLK "Upgrading IPs..."
    upgrade_ip [get_ips]
    update_compile_order -fileset sources_1
    lib_puts BLK "Successfully upgraded all IPs..."
  }
  
  switch -exact -- [lib_sys bd_ooc] {
    {None}          {}
    {Hierarchical}  {
                      lib_create_run_ip       [lib_sys bd_path] sources_1
                      lib_create_ip_stgy      [lib_sys bd_name] 40
                      dict with dict_prj dict_stgy {
                        if {[info exists synth_ip_custom]} {
                          lib_create_ip_custom    
                          lib_run_set_stgy    [dict get $dict_prj dict_stgy synth_ip_custom]
                        }
                        lib_run_synths        $synth_ip
                        lib_wait_runs         $synth_ip synth
                      }
                    }
    default         {lib_puts ERR "ERROR: The ooc option [lib_sys bd_ooc] is not supported yet..."
                     return -code 0 -level 2}               
  }
}

#****************************************************************
# create project and bd
#****************************************************************
proc lib_flow {} {
  global dict_prj
    lib_bd_logs
	
    lib_build_bd
    lib_puts FLW "Successfully created prj and bd!"

    lib_set_stgy  synth
    lib_set_stgy  impl
    dict with dict_prj dict_stgy {

    lib_create_run_synths     $synth

    lib_create_run_impls    $impl 
    lib_current_run         $impl implementation  

    lib_puts FLW "Successfully created runs!"
    }
}

#****************************************************************
# set xdc
#****************************************************************
dict set dict_prj dict_xdc [dict create                             \
  xf_pin        [dict create "PATH" [lib_sys xdc_dir]/pin.xdc     ] \
  xf_misc       [dict create "PATH" [lib_sys xdc_dir]/misc.xdc    ] \
  xf_timing     [dict create "PATH" [lib_sys xdc_dir]/timing.xdc  ] \
  xf_debug      [dict create "PATH" [lib_sys xdc_dir]/debug.xdc   ] \
  ]

#****************************************************************
# set ip selection
#   the key is dst, the value is src
#****************************************************************
dict set dict_prj dict_ip_sel [dict create    \
  {dpu}       {dpu}                           \
  ]
#****************************************************************
# set ps
#   for PSU__CRL_APB__PL?_REF_CTRL__SRCSEL: DPLL, IOPLL, RPLL
#   for GP ports
#       PSU__USE__M_AXI_GP0 :M_AXI_HPM0_FPD
#       PSU__USE__M_AXI_GP1 :M_AXI_HPM1_FPD
#       PSU__USE__M_AXI_GP2 :M_AXI_HPM0_LPD
#       PSU__USE__S_AXI_GP0 :S_AXI_HPC0_FPD
#       PSU__USE__S_AXI_GP1 :S_AXI_HPC1_FPD
#       PSU__USE__S_AXI_GP2 :S_AXI_HP0_FPD
#       PSU__USE__S_AXI_GP3 :S_AXI_HP1_FPD
#       PSU__USE__S_AXI_GP4 :S_AXI_HP2_FPD
#       PSU__USE__S_AXI_GP5 :S_AXI_HP3_FPD
#       PSU__USE__S_AXI_GP6 :S_AXI_LPD
#       PSU__USE__S_AXI_ACP :S_AXI_ACP_FPD
#       PSU__USE__S_AXI_ACE :S_AXI_ACE_FPD
#
#****************************************************************
dict set dict_prj dict_ip                                                 \
  ps  [dict create                                                        \
      PATH  zynq_ultra_ps_e                                               \
      NAME  {zynq_ultra_ps_e}                                             \
      VLNV  {}                                                            \
      BASE  [lib_sys tcl_base_dir]                                        \
      PROP  [dict create                                                  \
            "PSU__FPGA_PL0_ENABLE"                    1                   \
            "PSU__CRL_APB__PL0_REF_CTRL__SRCSEL"      IOPLL               \
            "PSU__CRL_APB__PL0_REF_CTRL__FREQMHZ"     [lib_param REG_CLK_MHz] \
                                                                          \
            "PSU__FPGA_PL1_ENABLE"                    1                   \
            "PSU__CRL_APB__PL1_REF_CTRL__SRCSEL"      RPLL                \
            "PSU__CRL_APB__PL1_REF_CTRL__FREQMHZ"     [lib_param HP_CLK_MHz]  \
                                                                          \
            "PSU__FPGA_PL2_ENABLE"                    0                   \
                                                                          \
            "PSU__FPGA_PL3_ENABLE"                    0                   \
                                                                          \
            "PSU__USE__FABRIC__RST"                   1                   \
            "PSU__NUM_FABRIC_RESETS"                  1                   \
                                                                          \
            "PSU__USE__M_AXI_GP2"                     1                   \
            "PSU__MAXIGP2__DATA_WIDTH"                32                  \
                                                                          \
            ]                                                             \
      CHKP  [dict create                                                  \
            "PSU__CRL_APB__PL0_REF_CTRL__ACT_FREQMHZ" 000.000             \
            "PSU__CRL_APB__PL1_REF_CTRL__ACT_FREQMHZ" 000.000             \
            "PSU__CRL_APB__PL2_REF_CTRL__ACT_FREQMHZ" 000.000             \
            "PSU__CRL_APB__PL3_REF_CTRL__ACT_FREQMHZ" 000.000             \
            ]                                                             \
      ]

#****************************************************************
# set hierarchy
#****************************************************************
dict set dict_prj dict_hier [dict create                  \
  h_dpu     [dict create "PATH" hier_dpu                ] \
  h_dpu_clk [dict create "PATH" hier_dpu/hier_dpu_clk   ] \
  h_dpu_ghp [dict create "PATH" hier_dpu/hier_dpu_ghp   ] \
  h_dpu_irq [dict create "PATH" hier_dpu/hier_dpu_irq   ] \
  ]

#****************************************************************
# set dpu_wrap
#****************************************************************
dict set dict_prj dict_verreg                           \
  info_sys              [dict create                    \
    DPU_PACK_ENA        {1}                             \
    DPU_SAXICLK_INDPD   [lib_param DPU_SAXICLK_INDPD]   \
    DPU_DSP48_LP_ENA    [lib_param DPU_CLK_GATING_ENA]  \
    DPU_IP_FOLDER       [lib_param DPU_IP_FOLDER]       \
    DPU_NUM             [lib_param DPU_NUM]             \
    DPU_EU_ENA          {1}                             \
    DPU_DSP48_VER       [expr {([lib_sys prj_family]=={zynq})?{DSP48E1}:{DSP48E2}}] \
    DPU_BKGRP           [expr {([string tolower [dict get $dict_prj dict_param  DPU_RAM_USAGE]]=={low})?{2}:{3}}] \
    DPU_LDP_ENA         {1}                             \
    DPU_HP_DATA_BW      [lib_param DPU_HP_DATA_BW]      \
    DPU_CLK_MHz         [lib_param DPU_CLK_MHz]         \
    HP_CC_EN            [lib_param DPU_HP_CC_EN]        \
    HIER_PATH_DPU       [lib_hier h_dpu]                \
    HIER_PATH_CLK       [lib_hier h_dpu_clk]            \
    HIER_PATH_GHP       [lib_hier h_dpu_ghp]            \
    HIER_PATH_IRQ       [lib_hier h_dpu_irq]            \
    DICT_IP_PS          {ps}                            \
    M_AXI_GP            "S_AXI_LPD"                     \
    M_AXI_HP            [list                           \
                        "S_AXI_HP0_FPD"                 \
                        "S_AXI_HP1_FPD"                 \
                        "S_AXI_HP2_FPD"                 \
                        "S_AXI_HP3_FPD"                 \
                        "S_AXI_HPC0_FPD"                \
                        "S_AXI_HPC1_FPD"                \
                        ]                               \
    M_AXI_CN_STGY       [lib_param CN_STGY]             \
    IRQ_MOD             {}                              \
    ]
dict set dict_prj dict_verreg info_ip         \
  dpu           [dict create                  \
    NAME        "dpu_eu"                      \
    PROP        [dict create                  \
                "VER_DPU_NUM"           [lib_param DPU_NUM]               \
                "ARCH"                  [lib_param DPU_ARCH]              \
				            "ARCH_IMG_BKGRP"        [expr {([string tolower [dict get $dict_prj dict_param  DPU_RAM_USAGE]]=={low})?{2}:{3}}]\
                "LOAD_AUGM"             [lib_param DPU_CHN_AUG_ENA]       \
                "DWCV_ENA"              [lib_param DPU_DWCV_ENA]          \
                "POOL_AVERAGE"          [lib_param DPU_AVG_POOL_ENA]      \
                "CONV_RELU_ADDON"       [lib_param DPU_CONV_RELU_TYPE]    \
                "CONV_WR_PARALLEL"      [lib_param DPU_CONV_WP]           \
																"SFM_ENA"               [lib_param DPU_SFM_NUM]           \
                "S_AXI_CLK_INDEPENDENT" [lib_param DPU_SAXICLK_INDPD]     \
                "CLK_GATING_ENA"        [lib_param DPU_CLK_GATING_ENA]    \
                "CONV_DSP_CASC_MAX"     [lib_param DPU_DSP48_MAX_CASC_LEN]\
                "CONV_DSP_ACCU_ENA"     [expr {([string tolower [dict get $dict_prj dict_param  DPU_DSP48_USAGE]]=={high})?{1}:{0}}]\
                "URAM_N_USER"           [lib_param DPU_URAM_PER_DPU]      \
                "TIMESTAMP_ENA"         [lib_param DPU_TIMESTAMP_ENA]     \
                ]                                                         \
    M_AXI_GP    [dict create                  \
                "M_AXI_INSTR" {M_AXI_GP0}     \
                ]                             \
    M_AXI_HP    {}                            \
    HP_DATA_BW  [lib_param DPU_HP_DATA_BW]    \
    ]
dict set dict_prj dict_verreg [lib_dpu_infos          [lib_dict_value dict_verreg]                            ]
dict set dict_prj dict_ip     [lib_dpu_ips_pack       [lib_dict_value dict_verreg] [lib_dict_value dict_ip  ] ]
dict set dict_prj dict_ip     [lib_dpu_ips_clk        [lib_dict_value dict_verreg] [lib_dict_value dict_ip  ] ]
dict set dict_prj dict_ip     [lib_dpu_ips_ghp        [lib_dict_value dict_verreg] [lib_dict_value dict_ip  ] ]
dict set dict_prj dict_ip     [lib_dpu_ips_ps         [lib_dict_value dict_verreg] [lib_dict_value dict_ip  ] ]
dict set dict_prj dict_ip     [lib_dpu_ips_irq        [lib_dict_value dict_verreg] [lib_dict_value dict_ip  ] ]
dict set dict_prj dict_pin    {}
dict set dict_prj dict_pin    [lib_dpu_pins_pack      [lib_dict_value dict_verreg] [lib_dict_value dict_pin ] ]
dict set dict_prj dict_pin    [lib_dpu_pins_clk       [lib_dict_value dict_verreg] [lib_dict_value dict_pin ] ]
dict set dict_prj dict_pin    [lib_dpu_pins_ghp       [lib_dict_value dict_verreg] [lib_dict_value dict_pin ] ]
dict set dict_prj dict_pin    [lib_dpu_pins_irq       [lib_dict_value dict_verreg] [lib_dict_value dict_pin ] ]
dict set dict_prj dict_cn     {}
dict set dict_prj dict_cn     [lib_dpu_cns_pack       [lib_dict_value dict_verreg] [lib_dict_value dict_cn  ] ]
dict set dict_prj dict_cn     [lib_dpu_cns_pack_clk   [lib_dict_value dict_verreg] [lib_dict_value dict_cn  ] ]
dict set dict_prj dict_cn     [lib_dpu_cns_ghp        [lib_dict_value dict_verreg] [lib_dict_value dict_cn  ] ]
dict set dict_prj dict_cn     [lib_dpu_cns_irq        [lib_dict_value dict_verreg] [lib_dict_value dict_cn  ] ]

#****************************************************************
# create pins
#   the prefix ph for hierarchy, pb for bd, pd for dpu_wrap
#   for CLASS:  PORT, PIN, INTF_PORT, INTF_PIN
#   for INTF
#       for MODE:   Master, Slave, System, MirroredMaster, MirroredSlave, MirroredSystem, Monitor
#   for PIN/PORT
#       for TYPE    : CLK, RST, CE, INTR, DATA, UNDEF(default, =OTHER)
#       for DIR     : I, O, IO
#       for VECTOR  : 0, 1
#       for PROP    : (optional)
#****************************************************************
# dict set dict_prj dict_pin [dict create  \
  # ph_dpu_GP0      [dict create  "CLASS" INTF_PIN  "PATH" [lib_hier h_dpu]/GP0         "MODE" Master "VLNV" {xilinx.com:interface:aximm_rtl:1.0} ]\
  # ph_dpu_PCLK     [dict create  "CLASS" PIN       "PATH" [lib_hier h_dpu]/PCLK        "TYPE" CLK    "DIR" I "VECTOR" 0  "FROM" 0  "TO" 0        ]\
  # ph_dpu_PRSTn    [dict create  "CLASS" PIN       "PATH" [lib_hier h_dpu]/PRSTn       "TYPE" RST    "DIR" I "VECTOR" 0  "FROM" 0  "TO" 0        ]\
  # ]

#****************************************************************
# set resets
#****************************************************************
dict set dict_prj dict_ip                 \
  ip_rst_reg    [dict create              \
                PATH  rst_gen_reg         \
                NAME  {proc_sys_reset}    \
                VLNV  {}                  \
                PROP  {}                  \
                ]
if { [lib_param DPU_HP_CC_EN] == {1} } {
  dict set dict_prj dict_ip                 \
    ip_rst_ghp    [dict create              \
                  PATH  rst_gen_ghp         \
                  NAME  {proc_sys_reset}    \
                  VLNV  {}                  \
                  PROP  {}                  \
                  ]
}

#****************************************************************
# set connect dict
#   top keys are
#     {PIN from PIN}, {PORT from PIN}, {PIN from PORT}, {PORT from PORT}
#     {PIN intf PIN}, {PORT intf PIN}, {PIN intf PORT}, {PORT intf PORT}
#   the key is dst, the value is src
#****************************************************************
dict set dict_prj dict_param  PIN_CLK_REG [lib_cell ps]/pl_clk0
if { [lib_param DPU_HP_CC_EN] == {1} } {
  dict set dict_prj dict_param  PIN_CLK_GHP [lib_cell ps]/pl_clk1
}

#****************************************************************
# set cn ps
#****************************************************************
if { [lib_param DPU_SAXICLK_INDPD] == {1} } {
  dict set dict_prj dict_cn  ps {PIN from PIN} [lib_cell ps]/maxihpm0_lpd_aclk  [lib_param PIN_CLK_REG]
} else {
  dict set dict_prj dict_cn  ps {PIN from PIN} [lib_cell ps]/maxihpm0_lpd_aclk  [lib_pin pd_ghp_GHP_CLK_O]
}

#****************************************************************
# set cn rst
#****************************************************************
dict set dict_prj dict_cn                                                               \
  cn_rst_reg  [dict create                                                              \
  {PIN from PIN}  [dict create                                                          \
                  [lib_cell ip_rst_reg]/slowest_sync_clk    [lib_param PIN_CLK_REG]     \
                  [lib_cell ip_rst_reg]/ext_reset_in        [lib_cell ps]/pl_resetn0    \
                  ]                                                                     \
  ]
if { [lib_param DPU_HP_CC_EN] == {1} } {
  dict set dict_prj dict_cn                                                               \
    cn_rst_ghp [dict create                                                               \
    {PIN from PIN}  [dict create                                                          \
                    [lib_cell ip_rst_ghp]/slowest_sync_clk    [lib_param PIN_CLK_GHP]     \
                    [lib_cell ip_rst_ghp]/ext_reset_in        [lib_cell ps]/pl_resetn0    \
                    ]                                                                     \
    ]
}

#****************************************************************
# set cn hier_dpu input
#****************************************************************
dict set dict_prj dict_cn                                                           \
  cn_h_dpu_clk [dict create                                                         \
  {PIN from PIN}  [dict create                                                      \
                  [lib_pin pd_clk_CLK]    [lib_param PIN_CLK_REG]                   \
                  [lib_pin pd_clk_RSTn]   [lib_cell ip_rst_reg]/peripheral_aresetn  \
                  ]                                                                 \
  ]
dict set dict_prj dict_cn cn_h_dpu {PIN intf PIN} [lib_pin pd_dpu_S_AXI]  [lib_cell ps]/M_AXI_HPM0_LPD
if { [lib_param DPU_SAXICLK_INDPD] == {1} } {
  dict set dict_prj dict_cn cn_h_dpu {PIN from PIN} [lib_pin pd_dpu_S_AXI_CLK]  [lib_param PIN_CLK_REG]
  dict set dict_prj dict_cn cn_h_dpu {PIN from PIN} [lib_pin pd_dpu_S_AXI_RSTn] [lib_cell ip_rst_reg]/peripheral_aresetn
}
if { [lib_param DPU_HP_CC_EN] == {1} } {
  dict set dict_prj dict_cn                                                   \
    cn_h_ghp [dict create                                                     \
    {PIN from PIN}  [dict create                                              \
                    [lib_pin pd_ghp_GHP_CLK_I]  [lib_param PIN_CLK_GHP]       \
                    [lib_pin pd_ghp_GHP_RSTn]   [lib_cell ip_rst_ghp]/peripheral_aresetn \
                    ]                                                         \
    ]
}

#****************************************************************
# set axi regs
#****************************************************************
dict set dict_prj dict_addr [dict create \
  ad_gp0 [dict create "REG" [lib_cell d_ip_dpu]/S_AXI/reg0      "RANGE" {16M}  "OFFSET" {0x8f000000} ]  \
  ]

#41d051e7f991c9dacbe8d368b85f57380b272bd2d7dc26e6c472a2d06e70908e
