set platform_name [lindex $argv 0]
puts "The platform name is \"$platform_name\""

set xsa_path [lindex $argv 1]
puts "The xsa path is \"$xsa_path\""

set sw_component [lindex $argv 2]
puts "The sw comp path is \"$sw_component\""

set output_path [lindex $argv 3]
puts "The output path is \"$output_path\""


platform create -name $platform_name -desc " A base platform targeting VCK190 which is the first Versal AI Core series evaluation kit, enabling designers to develop solutions using AI and DSP engines capable of delivering over 100X greater compute performance compared to current server class CPUs. This board includes 8GB of DDR4 UDIMM, 8GB LPDDR4 component, 400 AI engines, 1968 DSP engines, Dual-Core Arm® Cortex®-A72 and Dual-Core Cortex-R5. More information at https://www.xilinx.com/products/boards-and-kits/vck190.html" -hw $xsa_path/vck190_es1_base_trd_platform1.xsa -out $output_path -no-boot-bsp
domain create -name aiengine -os aie_runtime -proc {ai_engine}
domain config -pmcqemu-args $sw_component/platform/qemu/aie/pmc_args.txt
domain config -qemu-args $sw_component/platform/qemu/aie/qemu_args.txt
domain config -qemu-data $sw_component/platform/boot
domain create -name xrt -proc psv_cortexa72 -os linux -image $sw_component/platform/image
domain config -boot $sw_component/platform/boot
domain config -bif $sw_component/platform/boot/linux.bif

domain config -pmcqemu-args $sw_component/platform/qemu/lnx/pmc_args.txt
domain config -qemu-args $sw_component/platform/qemu/lnx/qemu_args.txt
domain config -qemu-data $sw_component/platform/boot

platform generate
