#!/usr/bin/env python3

# argument checking and help
import argparse
parser = argparse.ArgumentParser(
    description='Example of the LIBOR Market financial model running on a FPGA.')
required = parser.add_argument_group("required arguments")
required.add_argument("-x","--xclbin_file", dest="xclbin_filename", type=str, required=True,
    help="The model hardware bitstream xclbin filename. Build instructions for which are in '/<path to xf_fintech>/L2/tests/LMMEngineCap'")
required.add_argument("-c","--card",dest="card_type", type=str,required=True,
    help='Current supported Alveo cards are u200 and u250')
args = parser.parse_args()
# State test financial model and args entered
print("+--------------------------------------------------------------------")
print(parser.description)
print(args)
print("+--------------------------------------------------------------------")


#Ensure environmental variable i.e. paths are set to used the modules
import sys
# Check not using python 2
if sys.version.startswith("2"):
    sys.exit("Seem to be running with the no longer supported python 2 - require version 3")
from xf_fintech_python import DeviceManager, LMM
import numpy as np
from scipy.stats import norm
from os.path import exists

# Basic checking that the arguments are correct
if not (args.card_type == "u250" or args.card_type == "u200"):
    sys.exit("This version executes on either card type u200 or u250")
if not exists(args.xclbin_filename):
    sys.exit("Please check the supplied FPGA load filename - program does not see it")


def genSeeds():
    return list((np.random.rand(UN) * 1000).astype(int))

cVolas = [0.2366, 0.2487, 0.2573, 0.2564, 0.2476, 0.2376, 0.2252, 0.2246, 0.2223]
lRates = [0.0112, 0.0118, 0.0123, 0.0127, 0.0132, 0.0137, 0.0145, 0.0154, 0.0163, 0.0174]
noTenors = 10
noPaths = 10000
notional = 1e6
UN = 4


def capAnalytical(caprate, tau = 0.5):
    clet_prices = np.array([])
    for i in range(1, noTenors - 1):
        vol = cVolas[i - 1]
        L = lRates[i]
        T = tau * i
        d1 = (np.log(L / caprate) + 0.5 * vol * vol * T) / (vol * np.sqrt(T))
        d2 = d1 - (vol * np.sqrt(T))

        cr = (1/tau) * np.log(1 + np.sum(lRates[0:i+1]) / (i + 1) * tau)
        base_price = notional * tau * (L * norm.cdf(d1) - caprate * norm.cdf(d2))
        caplet_price = np.exp(-cr * tau * (i + 1)) * base_price
        np.append(clet_prices, caplet_price)
    return np.sum(clet_prices)


def capTest(caprate, paths):
    outPrice = []
    lmm.runCap(lRates, cVolas, genSeeds(), outPrice, noTenors, paths, 0.2, 1e6, caprate)
    expected = capAnalytical(caprate)
    print("\t[CPU]  Analytical Cap price: %.3f" % expected)
    print("\t[FPGA] LMM Cap price: %.3f" % outPrice[0])
    print("\t[FPGA] Runtime = %d" % lmm.lastruntime(), "us")
    diff = (outPrice[0] - expected) / expected * 100
    print("\t\tDiff = %.4f" % diff, "%")

# Program variables
deviceList = DeviceManager.getDeviceList(args.card_type) # Pass in the card type from the command line

if len(deviceList) == 0 : # Check at least one card found
    sys.exit(("Please check that you have a "+args.card_type+" card present and ready for use"))

# Identify which cards installed and choose the first available U200 card
print("Found there {0} device(s):".format(len(deviceList)))
for x in deviceList:
    print(x.getName())
chosenDevice = deviceList[0]
print("Choosing the first, ", str(chosenDevice), "\n")

# Selecting and loading into FPGA of chosen card the model to be used
lmm = LMM(args.xclbin_filename)

# Examples of possible operations for Cap pricing

print("\n-------------------------------")
print("        LMM Cap Pricing        ")
print("-------------------------------\n")

lmm.claimDeviceCap(chosenDevice)
print("CAP Example 1) K = 1.1%, 1K paths")
capTest(0.011, 1000)
print("CAP Example 2) K = 1.1%, 10K paths")
capTest(0.011, 10000)
print("CAP Example 3) K = 0.5%, 10K paths")
capTest(0.005, 10000)
print("CAP Example 4) K = 2.0%, 10K paths")
capTest(0.02, 10000)
lmm.releaseDevice()

print("End of example.\n")

sys.exit(0)

