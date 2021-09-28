#!/usr/bin/env python3

# argument checking and help
import argparse
parser = argparse.ArgumentParser(
    description='Example of the Merton76 financial model running on a FPGA.')
#formatter_class=RawTextHelpFormatter)

required = parser.add_argument_group("required arguments")
required.add_argument("-x","--xclbin_file", dest="xclbin_filename", type=str, required=True,
    help="The model hardware bitstream xclbin filename. Build instructions for which are in '/<path to xf_fintech>/L2/tests/M76Engine'")
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
from xf_fintech_python import DeviceManager, m76_input_data, m76
from os.path import exists

# Basic checking that the supplied arguments are correct
if not (args.card_type == "u250" or args.card_type == "u200"):
    sys.exit("This version executes on either card type u200 or u250")
if not exists(args.xclbin_filename):
    sys.exit("Please check the supplied FPGA load filename - program does not see it")

# State test financial model
#print("\nThe Merton76 financial model\n============================\n")

# Declaring Variables

# First the sample test data including expected result of the calculation
# order is StockPrice, StrikePrice, RiskFreeInterestRate,Volatility,time to vest (years),
#          kappa the expected[Y-1] where Y is the random variable, lambda - mean jump per unit time,
#          lambda - mean jump per unit time, and finally the expected result, which should be ignored
test_data_list = [
    [10.021600, 100.000000, 0.000000, 0.009900, 3.000000, 2.000000, -0.200000, 0.800000, 1.969068],
    [10.021600, 100.000000, 0.000000, 0.107500, 3.000000, 2.000000, -0.200000, 0.800000, 2.005603],
    [10.021600, 100.000000, 0.000000, 0.207900, 3.000000, 2.000000, -0.200000, 0.800000, 2.106863],
    [10.021600, 100.000000, 0.000000, 0.401500, 3.000000, 2.000000, -0.200000, 0.800000, 2.485681],
    [10.021600, 100.000000, 0.000000, 0.775200, 3.000000, 2.000000, -0.200000, 0.800000, 3.811954],
    [10.021600, 100.000000, 0.000000, 1.202000, 3.000000, 2.000000, -0.200000, 0.800000, 5.774945],
    [10.021600, 100.000000, 0.000000, 2.079800, 3.000000, 2.000000, -0.200000, 0.800000, 8.821378],
    [19.081700, 100.000000, 0.000000, 0.009900, 3.000000, 2.000000, -0.200000, 0.800000, 5.757761],
    [19.081700, 100.000000, 0.000000, 0.107500, 3.000000, 2.000000, -0.200000, 0.800000, 5.837310],
    [19.081700, 100.000000, 0.000000, 0.207900, 3.000000, 2.000000, -0.200000, 0.800000, 6.057548],
    [19.081700, 100.000000, 0.000000, 0.401500, 3.000000, 2.000000, -0.200000, 0.800000, 6.846772],
    [19.081700, 100.000000, 0.000000, 0.775200, 3.000000, 2.000000, -0.200000, 0.800000, 9.324169],
    [19.081700, 100.000000, 0.000000, 1.202000, 3.000000, 2.000000, -0.200000, 0.800000, 12.625823],
    [19.081700, 100.000000, 0.000000, 2.079800, 3.000000, 2.000000, -0.200000, 0.800000, 17.332820],
    [41.416200, 100.000000, 0.000000, 0.009900, 3.000000, 2.000000, -0.200000, 0.800000, 19.067555],
    [41.416200, 100.000000, 0.000000, 0.107500, 3.000000, 2.000000, -0.200000, 0.800000, 19.225610],
    [41.416200, 100.000000, 0.000000, 0.207900, 3.000000, 2.000000, -0.200000, 0.800000, 19.660236],
    [41.416200, 100.000000, 0.000000, 0.401500, 3.000000, 2.000000, -0.200000, 0.800000, 21.165239],
    [41.416200, 100.000000, 0.000000, 0.775200, 3.000000, 2.000000, -0.200000, 0.800000, 25.625022],
    [41.416200, 100.000000, 0.000000, 1.202000, 3.000000, 2.000000, -0.200000, 0.800000, 31.208559],
    [41.416200, 100.000000, 0.000000, 2.079800, 3.000000, 2.000000, -0.200000, 0.800000, 38.733838],
    [80.435600, 100.000000, 0.000000, 0.009900, 3.000000, 2.000000, -0.200000, 0.800000, 48.488604],
    [80.435600, 100.000000, 0.000000, 0.107500, 3.000000, 2.000000, -0.200000, 0.800000, 48.713268],
    [80.435600, 100.000000, 0.000000, 0.207900, 3.000000, 2.000000, -0.200000, 0.800000, 49.322781],
    [80.435600, 100.000000, 0.000000, 0.401500, 3.000000, 2.000000, -0.200000, 0.800000, 51.458253],
    [80.435600, 100.000000, 0.000000, 0.775200, 3.000000, 2.000000, -0.200000, 0.800000, 57.883240],
    [80.435600, 100.000000, 0.000000, 1.202000, 3.000000, 2.000000, -0.200000, 0.800000, 65.921074],
    [80.435600, 100.000000, 0.000000, 2.079800, 3.000000, 2.000000, -0.200000, 0.800000, 76.650626],
    [100.248000, 100.000000, 0.000000, 0.009900, 3.000000, 2.000000, -0.200000, 0.800000, 64.853552],
    [100.248000, 100.000000, 0.000000, 0.107500, 3.000000, 2.000000, -0.200000, 0.800000, 65.097010],
    [100.248000, 100.000000, 0.000000, 0.207900, 3.000000, 2.000000, -0.200000, 0.800000, 65.758732],
    [100.248000, 100.000000, 0.000000, 0.401500, 3.000000, 2.000000, -0.200000, 0.800000, 68.084161],
    [100.248000, 100.000000, 0.000000, 0.775200, 3.000000, 2.000000, -0.200000, 0.800000, 75.151581],
    [100.248000, 100.000000, 0.000000, 1.202000, 3.000000, 2.000000, -0.200000, 0.800000, 84.069285],
    [100.248000, 100.000000, 0.000000, 2.079800, 3.000000, 2.000000, -0.200000, 0.800000, 96.024020],
    [120.263000, 100.000000, 0.000000, 0.009900, 3.000000, 2.000000, -0.200000, 0.800000, 81.957940],
    [120.263000, 100.000000, 0.000000, 0.107500, 3.000000, 2.000000, -0.200000, 0.800000, 82.215043],
    [120.263000, 100.000000, 0.000000, 0.207900, 3.000000, 2.000000, -0.200000, 0.800000, 82.915088],
    [120.263000, 100.000000, 0.000000, 0.401500, 3.000000, 2.000000, -0.200000, 0.800000, 85.385085],
    [120.263000, 100.000000, 0.000000, 0.775200, 3.000000, 2.000000, -0.200000, 0.800000, 92.962528],
    [120.263000, 100.000000, 0.000000, 1.202000, 3.000000, 2.000000, -0.200000, 0.800000, 102.619584],
    [120.263000, 100.000000, 0.000000, 2.079800, 3.000000, 2.000000, -0.200000, 0.800000, 115.645671],
    [202.555000, 100.000000, 0.000000, 0.009900, 3.000000, 2.000000, -0.200000, 0.800000, 155.779813],
    [202.555000, 100.000000, 0.000000, 0.107500, 3.000000, 2.000000, -0.200000, 0.800000, 156.065895],
    [202.555000, 100.000000, 0.000000, 0.207900, 3.000000, 2.000000, -0.200000, 0.800000, 156.847402],
    [202.555000, 100.000000, 0.000000, 0.401500, 3.000000, 2.000000, -0.200000, 0.800000, 159.642731],
    [202.555000, 100.000000, 0.000000, 0.775200, 3.000000, 2.000000, -0.200000, 0.800000, 168.514010],
    [202.555000, 100.000000, 0.000000, 1.202000, 3.000000, 2.000000, -0.200000, 0.800000, 180.282417],
    [202.555000, 100.000000, 0.000000, 2.079800, 3.000000, 2.000000, -0.200000, 0.800000, 196.646414]
]


inputDataList =[]    # This is your list of input data , a list of lists
outputList = [] # This will be filled with a list of OptionPrices, one for each row of data input
numberOptions = len(test_data_list)    # Count the number rows of data in the test list

print("Re-ordering test example table.....found",numberOptions,"rows of data.")
for loop in range(0, numberOptions) :
    inputDataList.append(m76_input_data())
    inputDataList[loop].S = test_data_list[loop][0]
    inputDataList[loop].sigma = test_data_list[loop][3]
    inputDataList[loop].K = test_data_list[loop][1]
    inputDataList[loop].r = test_data_list[loop][2]
    inputDataList[loop].T = test_data_list[loop][4]
    inputDataList[loop].lamb = test_data_list[loop][5] # Not using lambda as it is a python keyword
    inputDataList[loop].kappa = test_data_list[loop][6]
    inputDataList[loop].delta = test_data_list[loop][7]
    #The final field is not passed to the FPGA. It was used to compare the result, as per the C++ example
    #This has now been determined to be of questionable accuracy. Please use your own methodology.
print("Example data placed into input data list\n")

deviceList = DeviceManager.getDeviceList(args.card_type) # Pass in the card type from the command line

if len(deviceList) == 0 : # Check at least one card found
    sys.exit(("Please check that you have a "+args.card_type+" card present and ready for use"))
          
print("Found these {0} device(s):".format(len(deviceList)))
for x in deviceList:
    print(x.getName())

print("Choosing the first suitable card\n")
chosenDevice = deviceList[0]


merton76 = m76(args.xclbin_filename) # This sets the xclbin file to load from the name supplied in the command line
merton76.claimDevice(chosenDevice)

merton76.run(inputDataList, outputList, numberOptions) #This is the call to process contents of dataList

#Format output to match the example in C++, simply to aid comparison of results
print("+-----+-----------+----------+-----------+---------+---------+---------+----------+---------+-----------+")
print("|Index| StockPrice|Volatility|StrikePrice|RiskFree |  Time   |  lambda |   kappa  |  delta  | OptionPice|")
print("+-----+-----------+----------+-----------+---------+---------+---------+----------+---------+-----------+")
for loop in range(0, numberOptions) :
    print("|%4d"%loop,"|%10.6f"%inputDataList[loop].S,"|%9.6f"%inputDataList[loop].sigma,
          "|%03.6f"%inputDataList[loop].K,"|%01.6f"%inputDataList[loop].r,"|%01.6f"%inputDataList[loop].T,
          "|%01.6f"%inputDataList[loop].lamb,"|%01.6f"%inputDataList[loop].kappa,"|%01.6f"%inputDataList[loop].delta,
          "|%10.6f"%outputList[loop],"|")
print("+-----+-----------+----------+-----------+---------+---------+---------+----------+---------+-----------+")

print("End of example.")

merton76.releaseDevice()

sys.exit(0)
