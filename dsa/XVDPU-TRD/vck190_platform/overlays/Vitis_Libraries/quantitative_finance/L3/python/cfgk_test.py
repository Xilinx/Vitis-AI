#!/usr/bin/env python3

# argument checking and help
import argparse
parser = argparse.ArgumentParser(
    description='Example of the Garman-Kohlhagen Closed Form financial model running on a FPGA.')
required = parser.add_argument_group("required arguments")
required.add_argument("-x","--xclbin_file", dest="xclbin_filename", type=str, required=True,
    help="The model hardware bitstream xclbin filename. Build instructions for which are in '/<path to xf_fintech>/L2/tests/GarmanKohlhagenEngine'")
required.add_argument("-c","--card",dest="card_type", type=str,required=True,
    help='Current supported Alveo cards are u200 and u250')
args = parser.parse_args()
# State financial model and args entered
print("+-----------------------------------------------------------------------------")
print(parser.description)
print(args)
print("+-----------------------------------------------------------------------------")

# Ensure environmental variables i.e. paths are set to the named the modules
import sys
# Check not using python 2
if sys.version.startswith("2"):
    sys.exit("Seem to be running with the no longer supported python 2 - require version 3")
from os.path import exists
from xf_fintech_python import DeviceManager, CFGarmanKohlhagen , OptionType

# Basic checking that the number of arguments are correct
if not (args.card_type == "u250" or args.card_type == "u200"):
    sys.exit("This version executes on either card type u200 or u250")
if not exists(args.xclbin_filename):
    sys.exit("Please check the supplied FPGA load filename - program does not see it")

# Declaring Variables
deviceList = DeviceManager.getDeviceList(args.card_type) # Pass in the card type from the command line

if len(deviceList) == 0 : # Check at least one card found
    sys.exit(("Please check that you have a "+args.card_type+" card present and ready for use"))

lastruntime = 0
numAssets = 0
# Inputs
stockPriceList = []
strikePriceList = []
volatilityList = []
timeToMaturityList = []
domesticRateList = []
foreignRateList = []
# Outputs - declaring them as empty lists
optionPriceList = []
deltaList = []
gammaList = []
vegaList = []
thetaList = []
rhoList = []


# Example financial data to test the module, same as used in the C++ example script
test_data_list = [
    [80, 80, 0.01, 0.01, 0.01, 0.3, 0.173888], 
    [80, 80, 0.01, 0.01, 0.01, 1, 0.315977],
    [80, 80, 0.01, 0.01, 0.01, 3, 0.536447], 
    [80, 80, 0.01, 0.01, 0.1, 0.3, 1.738665],
    [80, 80, 0.01, 0.01, 0.1, 1, 3.158466], 
    [80, 80, 0.01, 0.01, 0.1, 3, 5.357834],
    [80, 80, 0.01, 0.01, 1, 0.3, 17.174847], 
    [80, 80, 0.01, 0.01, 1, 1, 30.329180],
    [80, 80, 0.01, 0.01, 1, 3, 47.631312], 
    [80, 80, 0.01, 0.05, 0.01, 0.3, 0.002204],
    [80, 80, 0.01, 0.05, 0.01, 1, 0.000006], 
    [80, 80, 0.01, 0.05, 0.01, 3, 0.000000],
    [80, 80, 0.01, 0.05, 0.1, 0.3, 1.295909], 
    [80, 80, 0.01, 0.05, 0.1, 1, 1.787955],
    [80, 80, 0.01, 0.05, 0.1, 3, 1.827634], 
    [80, 80, 0.01, 0.05, 1, 0.3, 16.603307]
]


# Count the number of entries above - sets of data
numAssets = len(test_data_list)

# Populate the input lists from the above nested list
for loop in range(0, numAssets) :
    stockPriceList += [test_data_list[loop][0]]
    strikePriceList+= [test_data_list[loop][1]]
    volatilityList += [test_data_list[loop][4]]
    timeToMaturityList += [test_data_list[loop][5]]
    domesticRateList += [test_data_list[loop][2]]
    foreignRateList += [test_data_list[loop][3]]
    #This final field is not used, as the derivation method used to predict the expected result
    #has now been determined to be of questionable accuracy. Please use your own methodology.
    # Outputs - already declared those as empty lists


# Identify which cards are installed and choose the first available U250 card, as defined in deviceList above
print("Found these {0} device(s):".format(len(deviceList)))
for x in deviceList:
    print(x.getName())
print("Choosing the first suitable card\n")
chosenDevice = deviceList[0]

# Selecting and loading into FPGA on chosen card the financial model to be used
CFGarmanKohlhagen = CFGarmanKohlhagen(numAssets, args.xclbin_filename)   # warning the lower levels to accomodate at least this figure
CFGarmanKohlhagen.claimDevice(chosenDevice)
#Feed in the data and request the result
print("\nRunning...")
result = CFGarmanKohlhagen.run(stockPriceList, strikePriceList, volatilityList, timeToMaturityList, 
                    domesticRateList, foreignRateList, 
                    optionPriceList, deltaList, gammaList, vegaList, thetaList, rhoList, 
                    OptionType.Call, numAssets)
print("Done")
runtime = CFGarmanKohlhagen.lastruntime()

#Format output to match the example in C++, simply to aid comparison of results
print("+-------+-----------+----------------+--------------+---------------+---------------+---------------+")
print("| Index | Price     |     Delta      |     Gamma    |     Vega      |     Theta     |     Rho       |")
print("+-------+-----------+----------------+--------------+---------------+---------------+---------------+")
for loop in range(0, numAssets) :
    print(loop,"\t%9.5f"%optionPriceList[loop],"\t%9.5f"%deltaList[loop],"\t%9.5f"%gammaList[loop],"\t%9.5f"%vegaList[loop],"\t%9.5f"%thetaList[loop],"\t%9.5f"%rhoList[loop])


print("\nThis run took", str(runtime), "microseconds")

#Relinquish ownership of the card
CFGarmanKohlhagen.releaseDevice()

sys.exit(0)
