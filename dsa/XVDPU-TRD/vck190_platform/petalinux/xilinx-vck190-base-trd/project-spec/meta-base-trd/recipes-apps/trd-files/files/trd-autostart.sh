#!/bin/sh

# Source environment for init script
source /etc/profile

# Set console loglevel to KERN_INFO
echo "Setting console loglevel to 0 ..."
echo "0" > /proc/sys/kernel/printk

# Wake up monitor
/usr/bin/modetest -M xlnx > /dev/null

# update xclbin filename for VART
echo "Updating xclbin fileanme for /etc/vart.conf ..."
sed -i "s|dpu.xclbin|binary_container_1.xclbin|g" /etc/vart.conf
