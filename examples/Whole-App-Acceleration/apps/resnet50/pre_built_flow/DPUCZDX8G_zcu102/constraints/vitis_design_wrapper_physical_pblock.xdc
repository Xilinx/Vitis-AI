####################################################################################
# Constraints from file : 'pblock.xdc'
####################################################################################

current_instance -quiet
set_property HD.RECONFIGURABLE true [get_cells vitis_design_i/blobfromimage_accel_1]
create_pblock dpu_pblock
add_cells_to_pblock [get_pblocks dpu_pblock] [get_cells -quiet [list \
          vitis_design_i/DPUCZDX8G_1 \
          vitis_design_i/DPUCZDX8G_2 \
          vitis_design_i/GND \
          vitis_design_i/axcache_0xe \
          vitis_design_i/axi_ic_ps_e_S_AXI_HP0_FPD \
          vitis_design_i/axi_ic_ps_e_S_AXI_HP1_FPD \
          vitis_design_i/axi_ic_ps_e_S_AXI_HP2_FPD \
          vitis_design_i/axi_ic_ps_e_S_AXI_HPC0_FPD \
          vitis_design_i/axi_intc_0 \
          vitis_design_i/axi_intc_0_intr_1_interrupt_concat \
          vitis_design_i/axi_interconnect_lpd \
          vitis_design_i/axi_register_slice_0 \
          vitis_design_i/axi_vip_1 \
          vitis_design_i/axprot_0x2 \
          vitis_design_i/clk_wiz_0 \
          vitis_design_i/interconnect_axifull \
          vitis_design_i/interconnect_axihpm0fpd \
          vitis_design_i/interconnect_axilite \
          vitis_design_i/irq_const_tieoff \
          vitis_design_i/proc_sys_reset_1 \
          vitis_design_i/proc_sys_reset_2 \
          vitis_design_i/ps_e]]
resize_pblock [get_pblocks dpu_pblock] -add {SLICE_X29Y201:SLICE_X96Y239 SLICE_X77Y200:SLICE_X96Y200 SLICE_X29Y180:SLICE_X46Y200 SLICE_X84Y30:SLICE_X96Y199 SLICE_X36Y30:SLICE_X46Y179 SLICE_X36Y0:SLICE_X96Y29}
resize_pblock [get_pblocks dpu_pblock] -add {BIAS_X0Y0:BIAS_X0Y7}
resize_pblock [get_pblocks dpu_pblock] -add {BITSLICE_CONTROL_X0Y0:BITSLICE_CONTROL_X0Y31}
resize_pblock [get_pblocks dpu_pblock] -add {BITSLICE_RX_TX_X0Y0:BITSLICE_RX_TX_X0Y207}
resize_pblock [get_pblocks dpu_pblock] -add {BITSLICE_TX_X0Y0:BITSLICE_TX_X0Y31}
resize_pblock [get_pblocks dpu_pblock] -add {BUFCE_LEAF_X416Y12:BUFCE_LEAF_X527Y15}
resize_pblock [get_pblocks dpu_pblock] -add {BUFCE_ROW_X0Y72:BUFCE_ROW_X0Y95}
resize_pblock [get_pblocks dpu_pblock] -add {BUFCE_ROW_FSR_X104Y3:BUFCE_ROW_FSR_X129Y3}
resize_pblock [get_pblocks dpu_pblock] -add {BUFGCE_X0Y72:BUFGCE_X0Y95}
resize_pblock [get_pblocks dpu_pblock] -add {BUFGCE_DIV_X0Y12:BUFGCE_DIV_X0Y15}
resize_pblock [get_pblocks dpu_pblock] -add {BUFGCE_HDIO_X0Y2:BUFGCE_HDIO_X1Y3}
resize_pblock [get_pblocks dpu_pblock] -add {BUFGCTRL_X0Y24:BUFGCTRL_X0Y31}
resize_pblock [get_pblocks dpu_pblock] -add {DSP48E2_X6Y80:DSP48E2_X17Y95 DSP48E2_X16Y12:DSP48E2_X17Y79 DSP48E2_X6Y72:DSP48E2_X8Y79 DSP48E2_X7Y12:DSP48E2_X8Y71 DSP48E2_X7Y0:DSP48E2_X17Y11}
resize_pblock [get_pblocks dpu_pblock] -add {HARD_SYNC_X20Y6:HARD_SYNC_X25Y7}
resize_pblock [get_pblocks dpu_pblock] -add {HDIOBDIFFINBUF_X0Y12:HDIOBDIFFINBUF_X0Y23}
resize_pblock [get_pblocks dpu_pblock] -add {HDIOLOGIC_M_X0Y12:HDIOLOGIC_M_X0Y23}
resize_pblock [get_pblocks dpu_pblock] -add {HDIOLOGIC_S_X0Y12:HDIOLOGIC_S_X0Y23}
resize_pblock [get_pblocks dpu_pblock] -add {HDIO_BIAS_X0Y1:HDIO_BIAS_X0Y1}
resize_pblock [get_pblocks dpu_pblock] -add {HDIO_VREF_X0Y1:HDIO_VREF_X0Y1}
resize_pblock [get_pblocks dpu_pblock] -add {HPIOBDIFFINBUF_X0Y0:HPIOBDIFFINBUF_X0Y95}
resize_pblock [get_pblocks dpu_pblock] -add {HPIOBDIFFOUTBUF_X0Y0:HPIOBDIFFOUTBUF_X0Y95}
resize_pblock [get_pblocks dpu_pblock] -add {HPIO_VREF_SITE_X0Y0:HPIO_VREF_SITE_X0Y7}
resize_pblock [get_pblocks dpu_pblock] -add {IOB_X1Y0:IOB_X1Y207 IOB_X0Y156:IOB_X0Y193}
resize_pblock [get_pblocks dpu_pblock] -add {MMCM_X0Y3:MMCM_X0Y3}
resize_pblock [get_pblocks dpu_pblock] -add {PLL_X0Y6:PLL_X0Y7}
resize_pblock [get_pblocks dpu_pblock] -add {PLL_SELECT_SITE_X0Y0:PLL_SELECT_SITE_X0Y31}
resize_pblock [get_pblocks dpu_pblock] -add {RAMB18_X4Y80:RAMB18_X12Y95 RAMB18_X11Y72:RAMB18_X12Y79 RAMB18_X4Y72:RAMB18_X5Y79 RAMB18_X12Y12:RAMB18_X12Y71 RAMB18_X5Y12:RAMB18_X5Y71 RAMB18_X5Y0:RAMB18_X12Y11}
resize_pblock [get_pblocks dpu_pblock] -add {RAMB36_X4Y40:RAMB36_X12Y47 RAMB36_X11Y36:RAMB36_X12Y39 RAMB36_X4Y36:RAMB36_X5Y39 RAMB36_X12Y6:RAMB36_X12Y35 RAMB36_X5Y6:RAMB36_X5Y35 RAMB36_X5Y0:RAMB36_X12Y5}
resize_pblock [get_pblocks dpu_pblock] -add {RIU_OR_X0Y0:RIU_OR_X0Y15}
resize_pblock [get_pblocks dpu_pblock] -add {XIPHY_FEEDTHROUGH_X0Y0:XIPHY_FEEDTHROUGH_X0Y15}
resize_pblock [get_pblocks dpu_pblock] -add {CLOCKREGION_X0Y4:CLOCKREGION_X3Y6 CLOCKREGION_X0Y0:CLOCKREGION_X0Y3}
set_property EXCLUDE_PLACEMENT 1 [get_pblocks dpu_pblock]
# add_cells_to_pblock [get_pblocks dpu_pblock] [get_cells [list xilinx_zcu102_base_i/DPUCZDX8G_1 xilinx_zcu102_base_i/DPUCZDX8G_2 xilinx_zcu102_base_i/GND xilinx_zcu102_base_i/axcache_0xE xilinx_zcu102_base_i/axi_ic_ps_e_S_AXI_HP0_FPD xilinx_zcu102_base_i/axi_ic_ps_e_S_AXI_HP1_FPD xilinx_zcu102_base_i/axi_ic_ps_e_S_AXI_HP2_FPD xilinx_zcu102_base_i/axi_ic_ps_e_S_AXI_HPC0_FPD xilinx_zcu102_base_i/axi_intc_0 xilinx_zcu102_base_i/axi_intc_0_intr_1_interrupt_concat xilinx_zcu102_base_i/axi_interconnect_lpd xilinx_zcu102_base_i/axi_register_slice_0 xilinx_zcu102_base_i/axi_vip_1 xilinx_zcu102_base_i/axprot_val xilinx_zcu102_base_i/clk_wiz_0 xilinx_zcu102_base_i/interconnect_axifull xilinx_zcu102_base_i/interconnect_axihpm0fpd xilinx_zcu102_base_i/interconnect_axilite xilinx_zcu102_base_i/irq_const_tieoff xilinx_zcu102_base_i/proc_sys_reset_1 xilinx_zcu102_base_i/proc_sys_reset_2 xilinx_zcu102_base_i/ps_e]]
create_pblock pp_pipeline
add_cells_to_pblock [get_pblocks pp_pipeline] [get_cells -quiet [list vitis_design_i/blobfromimage_accel_1]]
resize_pblock [get_pblocks pp_pipeline] -add {SLICE_X47Y30:SLICE_X83Y200}
resize_pblock [get_pblocks pp_pipeline] -add {DSP48E2_X9Y12:DSP48E2_X15Y79}
resize_pblock [get_pblocks pp_pipeline] -add {RAMB18_X6Y12:RAMB18_X11Y79}
resize_pblock [get_pblocks pp_pipeline] -add {RAMB36_X6Y6:RAMB36_X11Y39}
set_property SNAPPING_MODE ON [get_pblocks pp_pipeline]
set_property IS_SOFT FALSE [get_pblocks pp_pipeline]

