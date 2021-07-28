# package RTL kernel IP to XO file
#
if { $::argc != 2 } {
    puts "ERROR: Program \"$::argv0\" requires 2 arguments!\n"
    puts "Usage: $::argv0  <TOP> <ACLK_FREQ>\n"
    exit
}
set TOP       [lindex $::argv 0]
set ACLK_FREQ [lindex $::argv 1]

source -notrace ./script/package_kernel_DPUCAHX8L_B.tcl

package_xo -force -xo_path ./DPUCAHX8L_B.xo -kernel_name DPUCAHX8L_B -ip_directory ./packaged_kernel/ip -kernel_xml ./src/kernel_DPUCAHX8L_B.xml
exit
