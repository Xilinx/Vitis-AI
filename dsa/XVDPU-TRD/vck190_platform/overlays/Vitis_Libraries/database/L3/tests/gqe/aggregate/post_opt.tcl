#
# Copyright 2019 Xilinx, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
add_cells_to_pblock pblock_dynamic_SLR1 [get_cells pfm_top_i/dynamic_region/gqeAggr_1/inst/writeTableAggr_*] -clear_locs
add_cells_to_pblock pblock_dynamic_SLR1 [get_cells pfm_top_i/dynamic_region/gqeAggr_1/inst/multi_dynamic_eval_wrapper_*] -clear_locs
add_cells_to_pblock pblock_dynamic_SLR1 [get_cells pfm_top_i/dynamic_region/gqeAggr_1/inst/scan_cols_wrapper_*] -clear_locs
add_cells_to_pblock pblock_dynamic_SLR1 [get_cells pfm_top_i/dynamic_region/gqeAggr_1/inst/filter_ongoing_*] -clear_locs
add_cells_to_pblock pblock_dynamic_SLR0 [get_cells pfm_top_i/dynamic_region/gqeAggr_1/inst/group_aggregate_wrapper_*] -clear_locs
