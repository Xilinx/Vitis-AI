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


if { $::argc != 4 } {
    puts "ERROR: Program \"$::argv0\" requires 4 arguments!\n"
    puts "Usage: $::argv0 <xoname> <krnl_name> <target> <device>\n"
    exit
}

set xoname    [lindex $::argv 0]
set krnl_name [lindex $::argv 1]
set target    [lindex $::argv 2]
set device    [lindex $::argv 3]
puts $xoname
set suffix "${krnl_name}_${target}_${device}"
if { [info exists ::env(DIR_PATH)] } {
    source -notrace $env(DIR_PRJ)/scripts/package_dpu_kernel.tcl
} else {
    source -notrace ./scripts/package_dpu_kernel.tcl
}

if {[file exists "${xoname}"]} {
    file delete -force "${xoname}"
}

if { [info exists ::env(DIR_PATH)] } {
    package_xo -xo_path ${xoname} -kernel_name ${krnl_name} -ip_directory ./packaged_kernel_${suffix} -kernel_xml $env(DIR_PRJ)/kernel_xml/dpu/kernel.xml
} else {
    package_xo -xo_path ${xoname} -kernel_name ${krnl_name} -ip_directory ./packaged_kernel_${suffix} -kernel_xml ./kernel_xml/dpu/kernel.xml
}
