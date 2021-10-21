#!/usr/bin/env python3

# argument checking and help
import argparse
parser = argparse.ArgumentParser(
    description='Example of the Population Markov Chain Monte Carlo financial model running on a FPGA.')
required = parser.add_argument_group("required arguments")
required.add_argument("-x","--xclbin_file", dest="xclbin_filename", type=str, required=True,
    help="The model hardware bitstream xclbin filename. Build instructions for which are in '/<path to xf_fintech>/L2/tests/PopMCMC'")
required.add_argument("-c","--card",dest="card_type", type=str,required=True,
    help='Current supported Alveo cards are u200 and u250')
args = parser.parse_args()
# State test financial model and args entered
print("+--------------------------------------------------------------------")
print(parser.description)
print(args)
print("+--------------------------------------------------------------------")

# Ensure environmental variables i.e. paths are set to used the modules
import sys
# Check not using python 2
if sys.version.startswith("2"):
    sys.exit("Seem to be running with the no longer supported python 2 - require version 3")
from os.path import exists
from xf_fintech_python import DeviceManager, PopMCMC

if not (args.card_type == "u250" or args.card_type == "u200"):
    sys.exit("This version executes on either card type u200 or u250")
if not exists(args.xclbin_filename):
    sys.exit("Please check the supplied FPGA load filename - program does not see it")

# Variables
deviceList = DeviceManager.getDeviceList(args.card_type) # Pass in the card type from the command line

if len(deviceList) == 0 : # Check at least one card found
    sys.exit(("Please check that you have a "+args.card_type+" card present and ready for use"))

outputData = []
lastruntime = 0


# Identify which cards installed and choose the first available U250 card
print("Found these {0} device(s):".format(len(deviceList)))
for x in deviceList:
    print(x.getName())
chosenDevice = deviceList[0]
print("Choosing the first, ",str(chosenDevice),"\n")


# Selecting and loading into FPGA on chosen card the financial model to be used
PopMCMC = PopMCMC(args.xclbin_filename)
PopMCMC.claimDevice(chosenDevice)

# Example
print("\nAn  example follows below,\n\nint samples, int burninSamples, double sigma, list output");
result = PopMCMC.run( 5000, 500, 0.4, outputData)
print("Output Result: num", len(outputData))
for i in range(len(outputData)):
                print(outputData[i])
                #i += 1

# Report runtime
runtime = PopMCMC.lastruntime()
print("This run took",str(runtime), "microseconds\n")


PopMCMC.releaseDevice()

sys.exit(0)

