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

#! /bin/bash

#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#Throw error message and exit
throw_error()
{
	echo "ERROR:prep_target.sh:$@" >&2
	exit 1
}

#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#Check command line options
target=""
xclbin=""
aie_path=""
linux_path=""
kernel=""
impl=""
while [ $# -ge 1 ]; do
	case "$1" in
		-h|-help|--help)
			echo "prep_target.sh -h|-help"
			echo "prep_target.sh -target hw_emu [-aie-work-dir <path>] [-linux-path <path>] -xclbin <xclbin> -kernel <kernel>"
			echo "prep_target.sh -target hw     [-aie-work-dir <path>] [-linux-path <path>] -impl <path>"
			echo ""
			echo "  -h|-help          Report usage information"
			echo ""
			echo "  -target           Specify flow (hw or hw_emu)"
			echo "  -xclbin           Specify xclbin file to update (needed for hw_emu only)"
			echo "  -aie-work-dir     Specify AIE directory (needed if design uses AIE; hw or hw_emu)"
			echo "  -linux-path       Specify petalinux path (needed if design uses ERT)"
			echo "  -kernel           Specify kernel name (needed for hw_emu only)"
			echo "  -impl             Specify implementation run directory (needed for hw only)"
			exit 0
			;;
		-target)
			shift
			target=$1
			;;
		-xclbin)
			shift
			xclbin=$1
			;;
		-aie-work-dir)
			shift
			aie_path=$1
			;;
		-linux-path)
			shift
			linux_path=$1
			;;
		-kernel)
			shift
			kernel=$1
			;;
		-impl)
			shift
			impl=$1
			;;
		*)
			throw_error "Unknown command line option $1; see prep_target.sh -help for usage"
			;;
	esac
	shift
done

[ "${target}" ] || throw_error "Missing argument; must specify target"

#Validate any aie_path, and convert to an absolute path
if [ "${aie_path}" ]; then
	[ -d "${aie_path}" ] || throw_error "Illegal aie_path; must point to an aiecompiler Work directory"
	aie_path="$(cd ${aie_path}; pwd -P)"
fi

#Validate any linux_path, and convert to an absolute path
if [ "${linux_path}" ]; then
	[ -d "${linux_path}" ] || throw_error "Illegal linux_path; must point to a linux image directory"
	linux_path="$(cd ${linux_path}; pwd -P)"
fi

[ "${XILINX_VITIS}" ] || throw_error "Missing environment variable; XILINX_VITIS must be defined and point to Vitis tools to use"

[ "${CARDANO_ROOT}" ] || export CARDANO_ROOT="${XILINX_VITIS}/cardano"

rm -fr .prep_target 1>/dev/null 2>/dev/null
mkdir -p .prep_target || exit 1

case "${target}" in

	#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
	#For hw_emu, we need to add any AIE code to the xclbin as an EMULATION_DATA section
	hw_emu)

		#Validate options
		[ "${xclbin}" ] || throw_error "Missing argument for hw_emu; must specify xclbin"
		[ "${kernel}" ] || throw_error "Missing argument for hw_emu; must specify kernel"

		#For hw_emu, we only need to update the xlcbin if we have AIE code to add
		if [ "${aie_path}" ]; then

			#Generate PDI for hw_emu
			echo "INFO:hw_emu:Generating new PDI..."
			mkdir -p .prep_target/pdi || exit 1
			export BOOTGEN_POST_PROCESSING=0
			qemu_path="$(cd ${XILINX_VITIS}/data/emulation/platforms/versal/sw/a72_standalone/qemu; pwd -P)"
			${XILINX_VITIS}/scripts/vitis/util/generate_pdi.sh -out-dir .prep_target/pdi -pmc-elf "${qemu_path}/plm.elf" -pmc-cdo "${qemu_path}/pmc_cdo.bin" -work-dir ${aie_path} -enable-aie-cores 1>/dev/null || exit 1

			#Need to create emulation data zip file
			echo "INFO:hw_emu:Generating emulation data..."
			mkdir -p .prep_target/${kernel} || exit 1
			cp ${aie_path}/arch/aieshim_solution.aiesol .prep_target/${kernel}/aieshim_solution.aiesol || exit 1
			cp .prep_target/pdi/boot_bh.bin             .prep_target/${kernel}/boot_bh.bin || exit 1
			cp .prep_target/pdi/qemu_qspi.bin           .prep_target/${kernel}/qemu_qspi.bin || exit 1
			(cd .prep_target; zip -r emulation_data.zip ${kernel}) || eixt 1

			#Need to delete any existing EMULATION_DATA section
			xclbinutil -q --force --input ${xclbin} --remove-section EMULATION_DATA --output .prep_target/new.xclbin 1>/dev/null 2>/dev/null
			#2>/dev/null
			if [ -f ".prep_target/new.xclbin" ]; then
				cp -f .prep_target/new.xclbin ${xclbin} || exit 1
			fi

			#Add new EMULATION_DATA section to xclbin
			echo "INFO:hw_emu:Adding emulation data to xclbin..."
			xclbinutil -q --force --input ${xclbin} --add-section EMULATION_DATA:RAW:.prep_target/emulation_data.zip --output .prep_target/new.xclbin 1>/dev/null || exit 1
			cp -f .prep_target/new.xclbin ${xclbin} || exit 1

		fi

		;;

	#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
	#For hw, we need to add any AIE code to the PDI
	hw)

		#Validate options
		[ "${impl}" ] || throw_error "Missing argument for hw; must specify impl"

		#Validate impl, and convert to an absolute path
		[ -d "${impl}" ] || throw_error "Illegal impl; must point to an implementation run directory"
		impl="$(cd ${impl}; pwd -P)"

		#Determine the AIE options for prep_target
		if [ "${aie_path}" ]; then
			aie_options="-aie-work-dir ${aie_path} -enable-aie-cores"
		else
			aie_options="-skip-aie-binaries"
		fi

		#Update pfm_top_wrapper.pdi
		echo "INFO:hw:Updating PDI..."
		[ -f ${impl}/pfm_top_wrapper.pdi ] || throw_error "Missing pfm_top_wrapper.pdi in implementation directory"
		${XILINX_VITIS}/scripts/vitis/util/prep_target -target hw -pdi ${impl}/pfm_top_wrapper.pdi ${aie_options} -boot-mode qspi -out-dir .prep_target || exit 1

		#Prep_target can't add linux ijn the manner we need, so will not modify the bif file it created
		if [ "${linux_path}" ]; then
			echo "INFO:hw:Adding petalinux..."
			gawk -v p=${linux_path} '$1=="}" { printf("  [bootloader] %s/versal_plm.elf\n",p); printf("  [raw, load=0x1000] %s/system.dtb\n",p); printf("  [raw, load=0x20000000] %s/boot.scr\n",p); printf("  [raw, load=0x30000000] %s/image.ub\n",p); printf("  %s/u-boot.elf\n",p); printf("  [destination_cpu=a72-0, exception_level=el-3, trustzone] %s/bl31.elf\n",p); } { print; }' .prep_target/boot_image.bif > .prep_target/boot_image.mod.bif || exit 1
			bootgen -arch versal -image .prep_target/boot_image.mod.bif -w -o .prep_target/BOOT.BIN || exit 1
		fi

		#Place update PDI in build directory
		rm -fr pfm_top_wrapper.pdi 1>/dev/null 2>/dev/null
		cp -f .prep_target/BOOT.BIN pfm_top_wrapper.pdi || exit 1

		;;

	#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
	*)
		throw_error "Illegal target ${taregt}; must be one of [hw hw_emu]"
		;;

esac

#Tidy temp area
rm -fr .prep_target 1>/dev/null 2>/dev/null

exit 0
