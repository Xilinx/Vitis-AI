#!/usr/bin/env python3

# Ensure environmental variables i.e. paths are set to used the modules
from xf_fintech_python import DeviceManager, FDHeston
import sys
import time

# Purpose of this file is to demonstrate the movement of ancillary functions away from each financial model
print("\n#######################################################")
print("# Ancillary python function demonstration test script #")
print("#######################################################")
print("This script is intended to run through the ancillary python api functions.")
print("These functions are used, as required, within the fin_tech example scripts.")
print("Some interact with one another; so it was decided to provide a single script")
print("which walks through an example of each function from within each api module.")
print("This should provide enough information, along with the example calculation scripts")
print("to enable you to make use of the python functions in your own scripts.")
print("\nNote: It assumes you have at least one Xilinx U series card installed")
print("      and you need to have the compiled pybind module in the same directory.")
print("\nWhen you are ready to continue please press the <enter> key") ; x = input()
print("Starting with the DeviceManager.hpp module\n------------------------------------------") ; time.sleep(2)
print("This has only one function - getDeviceList\n") ; time.sleep(2)
print("This function examines the machine to see which Xilinx FPGA cards are installed.")
print("The underlying C++ code does provide a degree of robustness as shown below ;")
print("This function uses the 'device.hpp' library, which has a list of Xilinx card types.")
print("Note: The underlying code produces some diagnostic information to stdout")

print("DeviceManager.getDeviceList() will list all Xilinx cards found")
deviceList = DeviceManager.getDeviceList("")

print("Found these {0} device(s):".format(len(deviceList))) # Print list
for x in deviceList:
    print(x.getName())


print('DeviceManager.getDeviceList("u200") will search for only that card type')
deviceList = DeviceManager.getDeviceList("u200")

print("Found these {0} device(s):".format(len(deviceList)))
for x in deviceList:
    print(x.getName())

print('DeviceManager.getDeviceList("madeupname") will search for only that card type')
deviceList = DeviceManager.getDeviceList("madeupname")

print("Found these {0} device(s):".format(len(deviceList)))
for x in deviceList:
    print(x.getName())


print("\nThat's the end of the Device Manager api example. Press <enter> key to continue")
x = input()
time.sleep(1)

print("End of script for the moment.")
time.sleep(0.1)

exit()
