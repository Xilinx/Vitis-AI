#!/usr/bin/env python3

# argument checking and help
import argparse
parser = argparse.ArgumentParser(
    description='Example of the Hull White Analytic Engine financial model running on a FPGA.')
required = parser.add_argument_group("required arguments")
required.add_argument("-x","--xclbin_file", dest="xclbin_filename", type=str, required=True,
    help="The model hardware bitstream xclbin filename. Build instructions for which are in '/<path to xf_fintech>/L2/tests/HullWhiteAnalyticEngine'")
required.add_argument("-c","--card",dest="card_type", type=str,required=True,
    help='Current supported Alveo cards are u200 and u250')
args = parser.parse_args()
# State financial model and args entered
print("+---------------------------------------------------------------------------")
print(parser.description)
print(args)
print("+---------------------------------------------------------------------------")

# Ensure environmental variables i.e. paths are set to the named the modules
import sys
# Check not using python 2
if sys.version.startswith("2"):
    sys.exit("Seem to be running with the no longer supported python 2 - require version 3")
from os.path import exists
from xf_fintech_python import DeviceManager, HullWhiteAnalytic
import random

# Basic checking that the arguments are correct
if not (args.card_type == "u250" or args.card_type == "u200"):
    sys.exit("This version executes on either card type u200 or u250")
if not exists(args.xclbin_filename):
    sys.exit("Please check the supplied FPGA load filename - program does not see it")

# Declaring Variables
deviceList = DeviceManager.getDeviceList(args.card_type) # Pass in the card type from the command line

if len(deviceList) == 0 : # Check at least one card found
    sys.exit(("Please check that you have a "+args.card_type+" card present and ready for use"))

lastruntime = 0
runtime = 0

# note these values should match the generated kernel
N_k0 = 16;
LEN = 16;
# Inputs - note that due to the random nature of the input data the output will also vary
a = 0.10;
sigma = 0.01;
ratesList = [0.0020, 0.0050, 0.0070, 0.0110, 0.0150, 0.0180, 0.0200, 0.0220, 0.0250, 0.0288, 0.0310, 0.0340, 0.0, 0.0 ,0.0, 0.0]
timesList = [0.25, 0.50, 0.75, 1.00, 1.50, 2.00, 3.00, 4.00, 5.00, 10.0, 20.0, 30.0, 0.0, 0.0, 0.0, 0.0]
tList = [random.uniform(0,15) for iteration in range(LEN)]
#print("tist is",tList)
TList = [(random.uniform(0,15)+1+tList[iteration]) for iteration in range(LEN)]
#print("TList is",TList)
# Output - declaring it as an empty list
PList= []

# Identify which cards are installed and choose the first available u200 card, as defined in deviceList above
print("Found these {0} device(s):".format(len(deviceList)))
for x in deviceList:
    print(x.getName())
print("Choosing the first suitable card\n")
chosenDevice = deviceList[0]

# Selecting and loading into FPGA on chosen card the financial model to be used
HullWhiteAnalytic = HullWhiteAnalytic(args.xclbin_filename)
HullWhiteAnalytic.claimDevice(chosenDevice)
#Feed in the data and request the result using tolerance method
print("\nRunning...")
result = HullWhiteAnalytic.run(a, sigma, timesList, ratesList, tList, TList, PList)
print("Done")
runtime = HullWhiteAnalytic.lastruntime()

# Display results
for i in range(len(PList)):
    print("HullWhite Spread [",i,"] = ",PList[i])
    i += 1

print("\nThis run took", str(runtime), "microseconds")

#Relinquish ownership of the card
HullWhiteAnalytic.releaseDevice()

sys.exit(0)
