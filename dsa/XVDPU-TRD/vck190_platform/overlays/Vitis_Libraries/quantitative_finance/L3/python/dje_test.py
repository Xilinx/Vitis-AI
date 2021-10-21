#!/usr/bin/env python3

# argument checking and help
import argparse
parser = argparse.ArgumentParser(
    description='Example of the Dow Jones MCEuropean financial model running on a FPGA.')
required = parser.add_argument_group("required arguments")
required.add_argument("-x","--xclbin_file", dest="xclbin_filename", type=str, required=True,
    help="The model hardware bitstream xclbin filename. Build instructions for which are in '/<path to xf_fintech>/L2/tests/MCEuropeanDowJonesEngine'")
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
from xf_fintech_python import DeviceManager, MCEuropeanDJE, OptionType
import array
from os.path import exists

# Basic checking that the number of arguments are correct
if not (args.card_type == "u250" or args.card_type == "u200"):
    sys.exit("This version executes on either card type u200 or u250")
if not exists(args.xclbin_filename):
    sys.exit("Please check the supplied FPGA load filename - program does not see it")

# Declaring Variables
deviceList = DeviceManager.getDeviceList(args.card_type) # Pass in the card type from the command line

if len(deviceList) == 0 : # Check at least one card found
    sys.exit(("Please check that you have a "+args.card_type+" card present and ready for use"))


outputResults = []
lastruntime = 0
# Example financial data to test the module
stockPriceList = [163.69, 117.38, 182.06, 347.82, 122.38, 116.65, 54.00, 50.64, 135.32, 49.53, 72.77,  187.86, 194.16, 131.49, 44.27,  134.14, 109.09, 199.87, 81.94,  124.86, 82.31,  42.67,  106.31, 148.56, 130.23, 244.16,  57.34,  163.21, 103.97, 50.81]
optionTypeList = [OptionType.Put] * len(stockPriceList)
strikePriceList = [0.0] * len(stockPriceList)
riskFreeRateList= [0.03] * len(stockPriceList)
volatilityList = [0.20] * len(stockPriceList)
dividendYieldList = [0.0] * len(stockPriceList)
timeToMaturityList = [1.0] * len(stockPriceList)
dowDivisor = 0.14748071991788
toleranceList = [0.02] * len(stockPriceList)
requiredSamplesList = [16383] * len(stockPriceList)
# with above data the result generated is 261.98479887630407



# Identify which cards are installed and choose the first available card
print("Found these {0} device(s):".format(len(deviceList)))
for x in deviceList:
    print(x.getName())
print("Choosing the first suitable card\n")
chosenDevice = deviceList[0]

# Selecting and loading into FPGA on chosen card the financial model to be used
mcEuropeanDJE = MCEuropeanDJE(args.xclbin_filename)
mcEuropeanDJE.claimDevice(chosenDevice)



#Feed in the data and request the result using tolerance method
print("\nRunning with tolerance...with arrays of asset values")
result = mcEuropeanDJE.run(optionTypeList, stockPriceList, strikePriceList, riskFreeRateList, dividendYieldList, volatilityList, timeToMaturityList, toleranceList, dowDivisor)
print("Option Price = {0}".format(result[1]))
runtime = mcEuropeanDJE.lastruntime()
print("This run took",str(runtime), "microseconds\n")

#Repeat this time using required number of samples method, and possible array of results, not actually needed for DJE
outputResults = [] # ensuring the list is empty, pybind offers an append not create.
print("Running with num samples...with arrays of asset values")
result = mcEuropeanDJE.run(optionTypeList, stockPriceList, strikePriceList, riskFreeRateList, dividendYieldList, volatilityList, timeToMaturityList, requiredSamplesList, dowDivisor, outputResults)
print("Output Result: num", len(outputResults))
for i in range(len(outputResults)):
                print(outputResults[i])
                i += 1
runtime = mcEuropeanDJE.lastruntime()
print("This run took", str(runtime), "microseconds")



#Relinquish ownership of the card
mcEuropeanDJE.releaseDevice()

sys.exit(0)

