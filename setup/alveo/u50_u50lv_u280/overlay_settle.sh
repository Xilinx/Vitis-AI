#!/bin/sh

print_usage () {
  echo "Usage: "
  echo "  overlay_settle.sh [DPU_type]"
  echo ""
  echo "  DPU_type can be V3E or V3ME"
  echo "  V3E represents DPUCAHX8H and V3ME represents DPUCAHX8L."
  echo "" 
}

if [ "$1" = "" ]
then
  print_usage
fi

case "$1" in
    V3E)
        echo "Settle XCLBIN for DPUCAHX8H."
        detected=0
        /opt/xilinx/xrt/bin/xbmgmt scan | grep xilinx_u50_
        if [ $? -eq 0 ]; then
          echo "U50 card detected."
          sudo cp ./alveo_xclbin-1.3.0/U50/6E300M/* /usr/lib
          detected=1
        fi

        /opt/xilinx/xrt/bin/xbmgmt scan | grep xilinx_u50lv_
        if [ $? -eq 0 ]; then
          echo "U50LV card detected."
          sudo cp ./alveo_xclbin-1.3.0/U50lv/10E275M/* /usr/lib
          detected=1
        fi

        /opt/xilinx/xrt/bin/xbmgmt scan | grep xilinx_u280_
        if [ $? -eq 0 ]; then
          echo "U280 card detected."
          sudo cp alveo_xclbin-1.3.0/U280/14E300M/* /usr/lib
          detected=1
        fi

        if [ $detected -eq 0 ]; then
          echo "ERROR! No compatible Alveo card (U50, U50LV, U280) detected!"
        else
          echo "Done."
        fi;;

    V3ME)
        echo "Settle XCLBIN for DPUCAHX8L."
        detected=0
        /opt/xilinx/xrt/bin/xbmgmt scan | grep xilinx_u50lv_
        if [ $? -eq 0 ]; then
          echo "U50LV card detected."
          sudo cp ./alveo_xclbin-1.3.0/U50lv-V3ME/1E300M/* /usr/lib
          detected=1
        fi

        /opt/xilinx/xrt/bin/xbmgmt scan | grep xilinx_u280_
        if [ $? -eq 0 ]; then
          echo "U280 card detected."
          sudo cp alveo_xclbin-1.3.0/U280-V3ME/2E300M/* /usr/lib
          detected=1
        fi

        if [ $detected -eq 0 ]; then
          echo "ERROR! No compatible Alveo card (U50LV, U280) detected!"
        else
          echo "Done."
        fi;;
    
    *) print_usage;;

esac
