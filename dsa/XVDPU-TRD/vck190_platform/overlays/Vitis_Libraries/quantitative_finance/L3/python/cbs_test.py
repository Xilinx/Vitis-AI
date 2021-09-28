#!/usr/bin/env python3

import argparse
parser = argparse.ArgumentParser(
    description='Example of the Black-Scholes Closed Form financial model running on a FPGA.')
required = parser.add_argument_group("required arguments")
required.add_argument("-x","--xclbin_file", dest="xclbin_filename", type=str, required=True,
    help="The model hardware bitstream xclbin filename. Build instructions for which are in '/<path to xf_fintech>/L2/tests/CFBlackScholes'")
required.add_argument("-c","--card",dest="card_type", type=str,required=True,
    help='Current supported Alveo cards are u200 and u250')
args = parser.parse_args()
# State test financial model and args entered
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
from xf_fintech_python import DeviceManager, CFBlackScholes, OptionType

if not (args.card_type == "u250" or args.card_type == "u200"):
    sys.exit("This version executes on either card type u200 or u250")
if not exists(args.xclbin_filename):
    sys.exit("Please check the supplied FPGA load filename - program does not see it")

# Declaring Variables
deviceList = DeviceManager.getDeviceList(args.card_type) # Pass in the card type from the command line

if len(deviceList) == 0 : # Check at least one card found
    sys.exit(("Please check that you have a "+args.card_type+" card present and ready for use"))

lastruntime = 0
# Example financial data to test the module as used in the C++ example script
numAssets = 100  # reduced from 100000 to 100 for clarity of script output - tested at 100000 samples
# Inputs
stockPriceList = [100.0] * numAssets
strikePriceList = [100.0] * numAssets
volatilityList = [0.1] * numAssets
riskFreeRateList= [0.025] * numAssets
timeToMaturityList = [1.0] * numAssets
# Outputs - declaring them as empty lists
optionPriceList = []
deltaList = []
gammaList = []
vegaList = []
thetaList = []
rhoList = []


# Identify which cards are installed and choose the first available card, as defined in deviceList above
print("Found these {0} device(s):".format(len(deviceList)))
for x in deviceList:
    print(x.getName())
print("Choosing the first suitable card\n")
chosenDevice = deviceList[0]

# Selecting and loading into FPGA on chosen card the financial model to be used
CFBlackScholes = CFBlackScholes(numAssets, args.xclbin_filename)   # warning the lower levels to accomodate at least this figure
CFBlackScholes.claimDevice(chosenDevice)
#Feed in the data and request the result using tolerance method
print("\nRunning...")
result = CFBlackScholes.run(stockPriceList, strikePriceList, volatilityList, riskFreeRateList, timeToMaturityList, 
                            optionPriceList, deltaList, gammaList, vegaList, thetaList, rhoList, OptionType.Put, numAssets)
print("Done")
runtime = CFBlackScholes.lastruntime()

#Format output to match the example in C++, simply to aid comparison of results
print("+-------+-----------+----------------+--------------+---------------+---------------+---------------+")
print("| Index | Price     |     Delta      |     Gamma    |     Vega      |     Theta     |     Rho       |")
print("+-------+-----------+----------------+--------------+---------------+---------------+---------------+")
for loop in range(0, numAssets) :
    print(loop,"\t%9.5f"%optionPriceList[loop],"\t%9.5f"%deltaList[loop],"\t%9.5f"%gammaList[loop],"\t%9.5f"%vegaList[loop],"\t%9.5f"%thetaList[loop],"\t%9.5f"%rhoList[loop])



print("\nThis run took", str(runtime), "microseconds")

#Relinquish ownership of the card
CFBlackScholes.releaseDevice()

sys.exit(0)
