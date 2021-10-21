#!/usr/bin/env python3

# argument checking and help
import argparse
parser = argparse.ArgumentParser(
    description='Example of the Credit Default Swap financial model running on a FPGA.')
required = parser.add_argument_group("required arguments")
required.add_argument("-x","--xclbin_file", dest="xclbin_filename", type=str, required=True,
    help="The model hardware bitstream xclbin filename. Build instructions for which are in '/<path to xf_fintech>/L2/tests/CreditDefaultSwapEngine'")
required.add_argument("-c","--card",dest="card_type", type=str,required=True,
    help='Current supported Alveo cards are u200 and u250')
args = parser.parse_args()
# State financial model and args entered
print("+--------------------------------------------------------------------")
print(parser.description)
print(args)
print("+--------------------------------------------------------------------")

# Ensure environmental variables i.e. paths are set to the named the modules
import sys
# Check not using python 2
if sys.version.startswith("2"):
    sys.exit("Seem to be running with the no longer supported python 2 - require version 3")
from os.path import exists
from xf_fintech_python import DeviceManager, CreditDefaultSwap

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

# Example financial data to test the module - same as used in the C++ test script
# Inputs
ratesIRList     = [0.0300, 0.0335, 0.0366, 0.0394, 0.0418, 0.0439, 0.0458, 0.0475, 0.0490, 0.0503, 0.0514,
                   0.0524, 0.0533, 0.0541, 0.0548, 0.0554, 0.0559, 0.0564, 0.0568, 0.0572, 0.0575]
timesIRList     = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 
                   5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0]
ratesHazardList = [0.005, 0.01, 0.01, 0.015, 0.010, 0.010]
timesHazardList = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0]
notionalList    = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
recoveryList    = [0.15, 0.67, 0.22, 0.01, 0.80, 0.99, 0.001, 0.44]
maturityList    = [2.0, 3.0, 4.0, 5.55, 6.33, 7.27, 8.001, 9.999]
frequencyList   = [4, 12, 2, 1, 12, 4, 1, 12]
# Outputs - declaring them as empty lists
CDSSpreadList = []

# Identify which cards are installed and choose the first available u200 card, as defined in deviceList above
print("Found these {0} device(s):".format(len(deviceList)))
for x in deviceList:
    print(x.getName())
print("Choosing the first suitable card\n")
chosenDevice = deviceList[0]

# Selecting and loading into FPGA on chosen card the financial model to be used
##CFB76 = CFB76(numAssets)   # warning the lower levels to accomodate at least this figure
CreditDefaultSwap = CreditDefaultSwap(args.xclbin_filename)
CreditDefaultSwap.claimDevice(chosenDevice)
#Feed in the data and request the result using tolerance method
print("\nRunning...")
result = CreditDefaultSwap.run(timesIRList, ratesIRList, timesHazardList, ratesHazardList, notionalList, 
                               recoveryList, maturityList, frequencyList, CDSSpreadList)
print("Done")
runtime = CreditDefaultSwap.lastruntime()

# Display results
for i in range(len(CDSSpreadList)):
    print("CDSSpread [",i,"] = ",CDSSpreadList[i])
    i += 1

print("\nThis run took", str(runtime), "microseconds")

#Relinquish ownership of the card
CreditDefaultSwap.releaseDevice()

sys.exit(0)
