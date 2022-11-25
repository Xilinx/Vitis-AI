# SLR pblocks
current_instance -quiet
create_pblock pblock_dynamic_SLR0
current_instance level0_i/ulp/SLR0/regslice_control_userpf/inst
add_cells_to_pblock [get_pblocks pblock_dynamic_SLR0] [get_cells *]
current_instance -quiet
current_instance level0_i/ulp/hmss_0/inst
add_cells_to_pblock [get_pblocks pblock_dynamic_SLR0] [get_cells -hierarchical -filter NAME=~*/path_28/slice0_28*]
add_cells_to_pblock [get_pblocks pblock_dynamic_SLR0] [get_cells -hierarchical -filter NAME=~*/path_28/interconnect0_28*]
add_cells_to_pblock [get_pblocks pblock_dynamic_SLR0] [get_cells -hierarchical -filter NAME=~*/path_0/slice1_0*]
add_cells_to_pblock [get_pblocks pblock_dynamic_SLR0] [get_cells -hierarchical -filter NAME=~*/path_0/interconnect1_0*]
add_cells_to_pblock [get_pblocks pblock_dynamic_SLR0] [get_cells -hierarchical -filter NAME=~*/path_10/slice2_10*]
add_cells_to_pblock [get_pblocks pblock_dynamic_SLR0] [get_cells -hierarchical -filter NAME=~*/path_10/interconnect2_10*]
add_cells_to_pblock [get_pblocks pblock_dynamic_SLR0] [get_cells -hierarchical -filter NAME=~*/path_11/slice3_11*]
add_cells_to_pblock [get_pblocks pblock_dynamic_SLR0] [get_cells -hierarchical -filter NAME=~*/path_11/interconnect3_11*]
add_cells_to_pblock [get_pblocks pblock_dynamic_SLR0] [get_cells -hierarchical -filter NAME=~*/path_8/slice4_8*]
add_cells_to_pblock [get_pblocks pblock_dynamic_SLR0] [get_cells -hierarchical -filter NAME=~*/path_8/interconnect4_8*]
add_cells_to_pblock [get_pblocks pblock_dynamic_SLR0] [get_cells -hierarchical -filter NAME=~*/path_4/slice5_4*]
add_cells_to_pblock [get_pblocks pblock_dynamic_SLR0] [get_cells -hierarchical -filter NAME=~*/path_4/interconnect5_4*]
add_cells_to_pblock [get_pblocks pblock_dynamic_SLR0] [get_cells -hierarchical -filter NAME=~*/path_5/slice6_5*]
add_cells_to_pblock [get_pblocks pblock_dynamic_SLR0] [get_cells -hierarchical -filter NAME=~*/path_5/interconnect6_5*]
add_cells_to_pblock [get_pblocks pblock_dynamic_SLR0] [get_cells -hierarchical -filter NAME=~*/path_1/slice7_1*]
add_cells_to_pblock [get_pblocks pblock_dynamic_SLR0] [get_cells -hierarchical -filter NAME=~*/path_1/interconnect7_1*]
add_cells_to_pblock [get_pblocks pblock_dynamic_SLR0] [get_cells -hierarchical -filter NAME=~*/path_12/slice8_12*]
add_cells_to_pblock [get_pblocks pblock_dynamic_SLR0] [get_cells -hierarchical -filter NAME=~*/path_12/interconnect8_12*]
add_cells_to_pblock [get_pblocks pblock_dynamic_SLR0] [get_cells -hierarchical -filter NAME=~*/path_13/slice9_13*]
add_cells_to_pblock [get_pblocks pblock_dynamic_SLR0] [get_cells -hierarchical -filter NAME=~*/path_13/interconnect9_13*]
add_cells_to_pblock [get_pblocks pblock_dynamic_SLR0] [get_cells -hierarchical -filter NAME=~*/path_9/slice10_9*]
add_cells_to_pblock [get_pblocks pblock_dynamic_SLR0] [get_cells -hierarchical -filter NAME=~*/path_9/interconnect10_9*]
add_cells_to_pblock [get_pblocks pblock_dynamic_SLR0] [get_cells -hierarchical -filter NAME=~*/path_6/slice11_6*]
add_cells_to_pblock [get_pblocks pblock_dynamic_SLR0] [get_cells -hierarchical -filter NAME=~*/path_6/interconnect11_6*]
add_cells_to_pblock [get_pblocks pblock_dynamic_SLR0] [get_cells -hierarchical -filter NAME=~*/path_7/slice12_7*]
add_cells_to_pblock [get_pblocks pblock_dynamic_SLR0] [get_cells -hierarchical -filter NAME=~*/path_7/interconnect12_7*]
add_cells_to_pblock [get_pblocks pblock_dynamic_SLR0] [get_cells -hierarchical -filter NAME=~*/path_2/slice13_2*]
add_cells_to_pblock [get_pblocks pblock_dynamic_SLR0] [get_cells -hierarchical -filter NAME=~*/path_2/interconnect13_2*]
add_cells_to_pblock [get_pblocks pblock_dynamic_SLR0] [get_cells -hierarchical -filter NAME=~*/path_3/slice14_3*]
add_cells_to_pblock [get_pblocks pblock_dynamic_SLR0] [get_cells -hierarchical -filter NAME=~*/path_3/interconnect14_3*]
add_cells_to_pblock [get_pblocks pblock_dynamic_SLR0] [get_cells -hierarchical -filter NAME=~*/path_22/slice15_22*]
add_cells_to_pblock [get_pblocks pblock_dynamic_SLR0] [get_cells -hierarchical -filter NAME=~*/path_22/interconnect15_22*]
add_cells_to_pblock [get_pblocks pblock_dynamic_SLR0] [get_cells -hierarchical -filter NAME=~*/path_14/slice16_14*]
add_cells_to_pblock [get_pblocks pblock_dynamic_SLR0] [get_cells -hierarchical -filter NAME=~*/path_14/interconnect16_14*]
add_cells_to_pblock [get_pblocks pblock_dynamic_SLR0] [get_cells -hierarchical -filter NAME=~*/path_20/slice17_20*]
add_cells_to_pblock [get_pblocks pblock_dynamic_SLR0] [get_cells -hierarchical -filter NAME=~*/path_20/interconnect17_20*]
add_cells_to_pblock [get_pblocks pblock_dynamic_SLR0] [get_cells -hierarchical -filter NAME=~*/hbm_reset_sync*SLR0*]
current_instance -quiet
add_cells_to_pblock [get_pblocks pblock_dynamic_SLR0] [get_cells {level0_i/ulp/SLR0 level0_i/ulp/dpu_0/inst/v3e_bd_i/axi_clock_converter_0 level0_i/ulp/dpu_0/inst/v3e_bd_i/axi_clock_converter_1 level0_i/ulp/dpu_0/inst/v3e_bd_i/axi_clock_converter_2 level0_i/ulp/dpu_0/inst/v3e_bd_i/axi_clock_converter_I0 level0_i/ulp/dpu_0/inst/v3e_bd_i/axi_clock_converter_W0 level0_i/ulp/dpu_0/inst/v3e_bd_i/axi_clock_converter_W1 level0_i/ulp/dpu_0/inst/v3e_bd_i/axi_clock_converter_csr level0_i/ulp/dpu_0/inst/v3e_bd_i/dpu_top_0 level0_i/ulp/dpu_1/inst/v3e_bd_i/axi_clock_converter_csr level0_i/ulp/hmss_0/inst/path_0 level0_i/ulp/hmss_0/inst/path_1 level0_i/ulp/hmss_0/inst/path_10 level0_i/ulp/hmss_0/inst/path_11 level0_i/ulp/hmss_0/inst/path_12 level0_i/ulp/hmss_0/inst/path_13 level0_i/ulp/hmss_0/inst/path_14 level0_i/ulp/hmss_0/inst/path_2 level0_i/ulp/hmss_0/inst/path_20 level0_i/ulp/hmss_0/inst/path_22 level0_i/ulp/hmss_0/inst/path_28 level0_i/ulp/hmss_0/inst/path_3 level0_i/ulp/hmss_0/inst/path_8 level0_i/ulp/hmss_0/inst/path_9}]
resize_pblock [get_pblocks pblock_dynamic_SLR0] -add {SLICE_X206Y0:SLICE_X232Y29 SLICE_X117Y60:SLICE_X145Y119 SLICE_X12Y60:SLICE_X30Y239}
resize_pblock [get_pblocks pblock_dynamic_SLR0] -add {BLI_HBM_APB_INTF_X30Y0:BLI_HBM_APB_INTF_X31Y0}
resize_pblock [get_pblocks pblock_dynamic_SLR0] -add {BLI_HBM_AXI_INTF_X30Y0:BLI_HBM_AXI_INTF_X31Y0}
resize_pblock [get_pblocks pblock_dynamic_SLR0] -add {CMACE4_X0Y0:CMACE4_X0Y1}
resize_pblock [get_pblocks pblock_dynamic_SLR0] -add {DSP48E2_X1Y18:DSP48E2_X3Y89 DSP48E2_X16Y18:DSP48E2_X19Y41 DSP48E2_X30Y0:DSP48E2_X31Y5}
resize_pblock [get_pblocks pblock_dynamic_SLR0] -add {LAGUNA_X2Y0:LAGUNA_X3Y119}
resize_pblock [get_pblocks pblock_dynamic_SLR0] -add {PCIE4CE4_X0Y1:PCIE4CE4_X0Y1}
resize_pblock [get_pblocks pblock_dynamic_SLR0] -add {RAMB18_X1Y24:RAMB18_X1Y95 RAMB18_X8Y24:RAMB18_X9Y47 RAMB18_X12Y0:RAMB18_X13Y11}
resize_pblock [get_pblocks pblock_dynamic_SLR0] -add {RAMB36_X1Y12:RAMB36_X1Y47 RAMB36_X8Y12:RAMB36_X9Y23 RAMB36_X12Y0:RAMB36_X13Y5}
resize_pblock [get_pblocks pblock_dynamic_SLR0] -add {URAM288_X2Y16:URAM288_X2Y31}
resize_pblock [get_pblocks pblock_dynamic_SLR0] -add {CLOCKREGION_X1Y2:CLOCKREGION_X6Y3 CLOCKREGION_X5Y1:CLOCKREGION_X6Y1 CLOCKREGION_X1Y1:CLOCKREGION_X3Y1 CLOCKREGION_X0Y0:CLOCKREGION_X6Y0}
set_property SNAPPING_MODE NESTED [get_pblocks pblock_dynamic_SLR0]
create_pblock pblock_dynamic_SLR1
current_instance level0_i/ulp/SLR1/regslice_control_userpf/inst
add_cells_to_pblock [get_pblocks pblock_dynamic_SLR1] [get_cells *]
current_instance -quiet
add_cells_to_pblock [get_pblocks pblock_dynamic_SLR1] [get_cells {level0_i/ulp/SLR1 level0_i/ulp/dpu_1/inst/v3e_bd_i/axi_clock_converter_0 level0_i/ulp/dpu_1/inst/v3e_bd_i/axi_clock_converter_1 level0_i/ulp/dpu_1/inst/v3e_bd_i/axi_clock_converter_2 level0_i/ulp/dpu_1/inst/v3e_bd_i/axi_clock_converter_I0 level0_i/ulp/dpu_1/inst/v3e_bd_i/axi_clock_converter_W0 level0_i/ulp/dpu_1/inst/v3e_bd_i/axi_clock_converter_W1 level0_i/ulp/dpu_1/inst/v3e_bd_i/dpu_top_0}]
resize_pblock [get_pblocks pblock_dynamic_SLR1] -add {SLICE_X220Y300:SLICE_X221Y359 SLICE_X117Y240:SLICE_X145Y299 SLICE_X12Y240:SLICE_X30Y299}
resize_pblock [get_pblocks pblock_dynamic_SLR1] -add {CMACE4_X0Y2:CMACE4_X0Y2}
resize_pblock [get_pblocks pblock_dynamic_SLR1] -add {CONFIG_SITE_X0Y1:CONFIG_SITE_X0Y1}
resize_pblock [get_pblocks pblock_dynamic_SLR1] -add {DSP48E2_X16Y90:DSP48E2_X19Y113 DSP48E2_X1Y90:DSP48E2_X3Y113}
resize_pblock [get_pblocks pblock_dynamic_SLR1] -add {LAGUNA_X16Y120:LAGUNA_X19Y239 LAGUNA_X2Y120:LAGUNA_X3Y239}
resize_pblock [get_pblocks pblock_dynamic_SLR1] -add {RAMB18_X8Y96:RAMB18_X9Y119 RAMB18_X1Y96:RAMB18_X1Y119}
resize_pblock [get_pblocks pblock_dynamic_SLR1] -add {RAMB36_X8Y48:RAMB36_X9Y59 RAMB36_X1Y48:RAMB36_X1Y59}
resize_pblock [get_pblocks pblock_dynamic_SLR1] -add {URAM288_X2Y64:URAM288_X2Y79}
resize_pblock [get_pblocks pblock_dynamic_SLR1] -add {CLOCKREGION_X0Y6:CLOCKREGION_X7Y7 CLOCKREGION_X0Y5:CLOCKREGION_X6Y5 CLOCKREGION_X5Y4:CLOCKREGION_X6Y4 CLOCKREGION_X1Y4:CLOCKREGION_X3Y4}
set_property SNAPPING_MODE NESTED [get_pblocks pblock_dynamic_SLR1]

# Level 1 pblock
# create_pblock pblock_dynamic_region
# add_cells_to_pblock [get_pblocks pblock_dynamic_region] [get_cells -quiet [list level0_i/ulp]]
# resize_pblock [get_pblocks pblock_dynamic_region] -add {SLICE_X220Y300:SLICE_X221Y359 SLICE_X117Y240:SLICE_X145Y299 SLICE_X117Y60:SLICE_X145Y119 SLICE_X206Y0:SLICE_X232Y29}
# resize_pblock [get_pblocks pblock_dynamic_region] -add {BLI_HBM_APB_INTF_X30Y0:BLI_HBM_APB_INTF_X31Y0}
# resize_pblock [get_pblocks pblock_dynamic_region] -add {BLI_HBM_AXI_INTF_X30Y0:BLI_HBM_AXI_INTF_X31Y0}
# resize_pblock [get_pblocks pblock_dynamic_region] -add {CONFIG_SITE_X0Y1:CONFIG_SITE_X0Y1}
# resize_pblock [get_pblocks pblock_dynamic_region] -add {DSP48E2_X16Y90:DSP48E2_X19Y113 DSP48E2_X16Y18:DSP48E2_X19Y41 DSP48E2_X30Y0:DSP48E2_X31Y5}
# resize_pblock [get_pblocks pblock_dynamic_region] -add {LAGUNA_X16Y120:LAGUNA_X19Y239}
# resize_pblock [get_pblocks pblock_dynamic_region] -add {RAMB18_X8Y96:RAMB18_X9Y119 RAMB18_X8Y24:RAMB18_X9Y47 RAMB18_X12Y0:RAMB18_X13Y11}
# resize_pblock [get_pblocks pblock_dynamic_region] -add {RAMB36_X8Y48:RAMB36_X9Y59 RAMB36_X8Y12:RAMB36_X9Y23 RAMB36_X12Y0:RAMB36_X13Y5}
# resize_pblock [get_pblocks pblock_dynamic_region] -add {URAM288_X2Y64:URAM288_X2Y79 URAM288_X2Y16:URAM288_X2Y31}
# resize_pblock [get_pblocks pblock_dynamic_region] -add {CLOCKREGION_X0Y6:CLOCKREGION_X7Y7 CLOCKREGION_X0Y5:CLOCKREGION_X6Y5 CLOCKREGION_X5Y4:CLOCKREGION_X6Y4 CLOCKREGION_X0Y4:CLOCKREGION_X3Y4 CLOCKREGION_X0Y2:CLOCKREGION_X6Y3 CLOCKREGION_X5Y1:CLOCKREGION_X6Y1 CLOCKREGION_X0Y1:CLOCKREGION_X3Y1 CLOCKREGION_X0Y0:CLOCKREGION_X6Y0}
# set_property CONTAIN_ROUTING 1 [get_pblocks pblock_dynamic_region]
# set_property EXCLUDE_PLACEMENT 1 [get_pblocks pblock_dynamic_region]
# set_property SNAPPING_MODE ON [get_pblocks pblock_dynamic_region]
# set_property IS_SOFT FALSE [get_pblocks pblock_dynamic_region]

# User Generated physical constraints 

create_pblock pblock_pp_pipeline
add_cells_to_pblock [get_pblocks pblock_pp_pipeline] [get_cells -quiet [list level0_i/ulp/blobfromimage_accel_1]]
resize_pblock [get_pblocks pblock_pp_pipeline] -add {SLICE_X0Y60:SLICE_X11Y299}
resize_pblock [get_pblocks pblock_pp_pipeline] -add {BUFG_GT_X0Y24:BUFG_GT_X0Y119}
resize_pblock [get_pblocks pblock_pp_pipeline] -add {BUFG_GT_SYNC_X0Y15:BUFG_GT_SYNC_X0Y74}
resize_pblock [get_pblocks pblock_pp_pipeline] -add {DSP48E2_X0Y18:DSP48E2_X0Y113}
resize_pblock [get_pblocks pblock_pp_pipeline] -add {GTYE4_CHANNEL_X0Y4:GTYE4_CHANNEL_X0Y19}
resize_pblock [get_pblocks pblock_pp_pipeline] -add {GTYE4_COMMON_X0Y1:GTYE4_COMMON_X0Y4}
resize_pblock [get_pblocks pblock_pp_pipeline] -add {LAGUNA_X0Y0:LAGUNA_X1Y239}
resize_pblock [get_pblocks pblock_pp_pipeline] -add {RAMB18_X0Y24:RAMB18_X0Y119}
resize_pblock [get_pblocks pblock_pp_pipeline] -add {RAMB36_X0Y12:RAMB36_X0Y59}
set_property EXCLUDE_PLACEMENT 1 [get_pblocks pblock_pp_pipeline]
set_property SNAPPING_MODE ON [get_pblocks pblock_pp_pipeline]
set_property IS_SOFT FALSE [get_pblocks pblock_pp_pipeline]

# User Generated miscellaneous constraints 

set_property PARENT pblock_dynamic_region [get_pblocks pblock_pp_pipeline]
set_property PARENT pblock_dynamic_region [get_pblocks pblock_dynamic_SLR0]
set_property PARENT pblock_dynamic_region [get_pblocks pblock_dynamic_SLR1]

