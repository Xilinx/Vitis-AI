# /*
# * Copyright 2019 Xilinx Inc.
# *
# * Licensed under the Apache License, Version 2.0 (the "License");
# * you may not use this file except in compliance with the License.
# * You may obtain a copy of the License at
# *
# *    http://www.apache.org/licenses/LICENSE-2.0
# *
# * Unless required by applicable law or agreed to in writing, software
# * distributed under the License is distributed on an "AS IS" BASIS,
# * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# * See the License for the specific language governing permissions and
# * limitations under the License.
# */


if { $::argc != 6 } {
    puts "ERROR: Program \"$::argv0\" requires 6 arguments!\n"
    puts "Usage: $::argv0 <top_dir> <xoname> <krnl_name> <target> <xpfm_path> <device>\n"
    exit
}

set top_dir   [lindex $::argv 0]
set xoname    [lindex $::argv 1]
set krnl_name [lindex $::argv 2]
set target    [lindex $::argv 3]
set xpfm_path [lindex $::argv 4]
set device    [lindex $::argv 5]

set suffix "${krnl_name}_${target}_${device}"

variable script_dir [file dirname [ file normalize [ info script ]]]
puts $script_dir
source -notrace $script_dir/package_dpu_kernel.tcl

if {[file exists "${xoname}"]} {
    file delete -force "${xoname}"
}

package_xo -xo_path ${xoname} -kernel_name ${krnl_name} -ip_directory ./packaged_kernel_${suffix} -kernel_xml $top_dir/src/prj/Vitis/kernel_xml/dpu/kernel.xml
