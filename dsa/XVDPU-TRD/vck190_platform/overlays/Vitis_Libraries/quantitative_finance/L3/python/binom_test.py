#!/usr/bin/env python3

# argument checking and help
import argparse
parser = argparse.ArgumentParser(
    description='Example of the Binomial financial model running on a FPGA.')
required = parser.add_argument_group("required arguments")
required.add_argument("-x","--xclbin_file", dest="xclbin_filename", type=str, required=True,
    help="The model hardware bitstream xclbin filename. Build instructions for which are in '/<path to xf_fintech>/L2/tests/BinomialTreeEngine'")
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
from xf_fintech_python import DeviceManager, BinomialTreeInputDataTypeDouble, BinomialTree, OptionType
import array

# Basic checking that the number of arguments are correct
if not (args.card_type == "u250" or args.card_type == "u200"):
    sys.exit("This version executes on either card type u200 or u250")
if not exists(args.xclbin_filename):
    sys.exit("Please check the supplied FPGA load filename - program does not see it")

deviceList = DeviceManager.getDeviceList(args.card_type) # Pass in the card type from the command line

if len(deviceList) == 0 : # Check at least one card found
    sys.exit(("Please check that you have a "+args.card_type+" card present and ready for use"))

print("Found these {0} device(s):".format(len(deviceList)))
for x in deviceList:
    print(x.getName())

print("Choosing the first suitable card\n")
chosenDevice = deviceList[0]


dataList = []
numberOptions = 8  #Note must be in multiples of 8

for i in range(numberOptions):
    dataList.append(BinomialTreeInputDataTypeDouble())
    dataList[i].S = 110.0
    dataList[i].K = 100.0 + i
    dataList[i].T = 1.0
    dataList[i].rf = 0.05
    dataList[i].V = 0.2
    dataList[i].q = 0
    dataList[i].N = 1024

bto = BinomialTree(args.xclbin_filename)

bto.claimDevice(chosenDevice)

outputlist = []

bto.run(dataList, outputlist, BinomialTree.OptionTypeAmericanPut)

print("\nMultiple [",numberOptions,"] Options using American Put")
for i in range(len(outputlist)):
    print(outputlist[i])
# Expect around 2.987749, 3.271813, 3.573721, 3.896820, 4.238249, 4.598072, 4.976364, 5.377398

outputlist = []

bto.run(dataList, outputlist, BinomialTree.OptionTypeAmericanCall)

print("\nMultiple [",numberOptions,"] Options using American Call")
for i in range(len(outputlist)):
    print(outputlist[i])
# Expect around 17.66420, 16.97225, 16.29594, 15.63922, 14.99773, 14.37137, 13.76001, 13.16996

bto.releaseDevice()

print("\nEnd of example for Binomial")
sys.exit(0)

