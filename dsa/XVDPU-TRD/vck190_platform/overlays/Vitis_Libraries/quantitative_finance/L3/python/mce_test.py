#!/usr/bin/env python3

# argument checking and help
import argparse
parser = argparse.ArgumentParser(
    description='Example of the MCEuropean financial model running on a FPGA.')
#formatter_class=RawTextHelpFormatter)

required = parser.add_argument_group("required arguments")
required.add_argument("-x","--xclbin_file", dest="xclbin_filename", type=str, required=True,
    help="The model hardware bitstream xclbin filename. Build instructions for which are in '/<path to xf_fintech>/L2/tests/MCEuropean'")
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
from xf_fintech_python import DeviceManager, MCEuropean, OptionType
import array

# Basic checking that the number of arguments are correct
if not (args.card_type == "u250" or args.card_type == "u200"):
    sys.exit("This version executes on either card type u200 or u250")
if not exists(args.xclbin_filename):
    sys.exit("Please check the supplied FPGA load filename - program does not see it")

# Variables
deviceList = DeviceManager.getDeviceList(args.card_type) # Pass in the card type from the command line

if len(deviceList) == 0 : # Check at least one card found
    sys.exit(("Please check that you have a "+args.card_type+" card present and ready for use"))

lastruntime = 0

# Identify which cards installed and choose the first available U250 card
print("Found these {0} device(s):".format(len(deviceList)))
for x in deviceList:
    print(x.getName())
chosenDevice = deviceList[0]
print("Choosing the first, ",str(chosenDevice),"\n")


# Selecting and loading into FPGA on chosen card the financial model to be used
mcEuropean = MCEuropean(args.xclbin_filename)
mcEuropean.claimDevice(chosenDevice)

# Examples of possible operations
print("\nA few examples follow below\n\nRunning with tolerance...single asset value");
result = mcEuropean.run(OptionType.Put, 36.0, 40.0, 0.06, 0.0, 0.20, 1.0, 0.02)
print("Option Price = {0}".format(result[1]))
runtime = mcEuropean.lastruntime()
print("This run took",str(runtime), "microseconds\n")

#print("Running with tolerance...single asset value");
#result = mcEuropean.run(OptionType.Put, 36.0, 40.0, 0.06, 0.0, 0.20, 1.0, 0.02)
#print("Output Results: num", len(outputResults))
# IH - to test - despite this only returning one reather than many values should still work
#for i in range(len(outputResults)):
#                print(outputResults[i])
#                i += 1


print("Running for required num samples...single asset value");
result = mcEuropean.run(OptionType.Put, 36.0, 40.0, 0.06, 0.0, 0.20, 1.0, 16383)
print("Option Price = {0}".format(result[1]))
runtime = mcEuropean.lastruntime()
print("This run took",str(runtime), "microseconds\n")

##print("Running for required num samples...single asset value");
##result = mcEuropean.run(OptionType.Put, 36.0, 40.0, 0.06, 0.0, 0.20, 1.0, 16383)
##print("Output Results: num", len(outputResults))
##for i in range(len(outputResults)):
##                print(outputResults[i])
##                i += 1


##initialise some arrays of asset data that we will use for out next set of calls...
optionTypeList     	= [OptionType.Put, OptionType.Put, OptionType.Put	]
stockPriceList     	= [36.0, 		   38.0, 		   40.0				]
strikePriceList    	= [40.0, 		   42.0, 		   44.0				]
riskFreeRateList   	= [0.06, 		   0.06, 		   0.06				]
dividendYieldList  	= [0.0, 		   0.0, 		   0.0				]
volatilityList     	= [0.20, 		   0.20, 		   0.20				]
timeToMaturityList 	= [1.0, 		   1.0, 		   1.0				]
toleranceList      	= [0.02, 		   0.02, 		   0.02				]
	                                                   
requiredSamplesList	= [16383,		   16383,          16383			]


print("Running with tolerance...with arrays of asset values");
result = mcEuropean.run(optionTypeList, stockPriceList, strikePriceList, riskFreeRateList, dividendYieldList, volatilityList, timeToMaturityList, toleranceList)
print("Option Prices = {0}".format(result[1]))
runtime = mcEuropean.lastruntime()
print("This run took",str(runtime), "microseconds\n")

#outputResults = []
#print("Running with tolerance...with arrays of asset values");
#result = mcEuropean.run(optionTypeList, stockPriceList, strikePriceList, riskFreeRateList, dividendYieldList, volatilityList, timeToMaturityList, toleranceList, outputResults)
#print("Output Results: num", len(outputResults))
#for i in range(len(outputResults)):
#                print(outputResults[i])
#                i += 1
#runtime = mcEuropean.lastruntime()


#print("Running with num samples...with arrays of asset values")
#result = mcEuropean.run(optionTypeList, stockPriceList, strikePriceList, riskFreeRateList, dividendYieldList, volatilityList, timeToMaturityList, requiredSamplesList)
#print("Option Prices = {0}".format(result[1]))

outputResults = []
print("Running with num samples...with arrays of asset values")
result = mcEuropean.run(optionTypeList, stockPriceList, strikePriceList, riskFreeRateList, dividendYieldList, volatilityList, timeToMaturityList, requiredSamplesList, outputResults)
print("Output Results: num", len(outputResults))
for i in range(len(outputResults)):
                print(outputResults[i])
                i += 1
runtime = mcEuropean.lastruntime()
print("This run took", str(runtime), "microseconds")

mcEuropean.releaseDevice()

sys.exit(0)

