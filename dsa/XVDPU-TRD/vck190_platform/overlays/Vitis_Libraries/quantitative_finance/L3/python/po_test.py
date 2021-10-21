#!/usr/bin/env python3

# argument checking and help
import argparse
parser = argparse.ArgumentParser(
    description='Example of the Portfolio Optimisation financial model running on a FPGA.')
required = parser.add_argument_group("required arguments")
required.add_argument("-x","--xclbin_file", dest="xclbin_filename", type=str, required=True,
    help="The model hardware bitstream xclbin filename. Build instructions for which are in '/<path to xf_fintech>/L2/tests/PortfolioOptimisation'")
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
from xf_fintech_python import DeviceManager, portfolio_optimisation

# Basic checking that the arguments are correct
if not (args.card_type == "u250" or args.card_type == "u200"):
    sys.exit("This version executes on either card type u200 or u250")
if not exists(args.xclbin_filename):
    sys.exit("Please check the supplied FPGA load filename - program does not see it")

# Declaring Variables
deviceList = DeviceManager.getDeviceList(args.card_type) # Pass in the card type from the command line

if len(deviceList) == 0 : # Check at least one card found
    sys.exit(("Please check that you have a "+args.card_type+" card present and ready for use"))

# Example financial data to test the module - same as used in the C++ test script
# Inputs
testDataList     = [
    37.57,  38.74,  41.52,  43.47,  43.65,  41.17,  42.35,  46.84,  51.38,  51.60,  51.99,  47.29,  48.08,
    49.56,  52.94,  52.71,  50.28,  53.47,  55.79,  55.52,  52.13,  53.33,  56.46,  52.67,  47.87,  50.00,
    48.83,  54.05,  53.11,  50.86,  52.41,  53.19,  54.62,  59.45,  58.69,  58.68,  60.53,  63.25,  66.92,
    63.92,  62.97,  66.66,  70.36,  71.76,  74.90,  76.00,  74.51,  71.51,  74.06,  74.46,  76.63,  80.39,
    83.13,  85.26,  89.74,  87.16,  92.16,  95.52,  100.33, 99.05,  99.99,

    84.74,  94.76,  97.03,  108.33, 104.10, 94.09,  90.65,  101.65, 103.53, 93.93,  116.25, 98.03,  104.53,
    122.29, 148.39, 166.72, 178.37, 154.79, 128.73, 75.08,  35.68,  29.69,  36.33,  29.32,  19.38,  20.81,
    26.15,  33.63,  35.26,  39.22,  43.24,  43.83,  34.07,  44.17,  54.52,  43.95,  52.42,  62.90,  54.12,
    46.79,  38.21,  43.94,  42.16,  43.50,  42.39,  48.28,  58.02,  57.28,  57.15,  53.62,  47.43,  45.89,
    45.82,  39.80,  30.01,  21.94,  25.27,  27.26,  26.42,  30.14,  31.01,

    44.84,  46.63,  47.10,  53.19,  55.32,  54.10,  58.99,  69.82,  71.24,  66.07,  69.23,  59.42,  68.04,
    73.58,  80.14,  89.35,  89.45,  78.93,  70.99,  44.80,  23.82,  21.48,  22.74,  20.87,  18.02,  18.69,
    21.99,  31.13,  31.04,  33.82,  33.61,  35.03,  32.09,  37.23,  43.37,  36.67,  36.40,  41.82,  36.99,
    29.07,  25.65,  29.43,  27.96,  31.81,  31.24,  30.73,  36.99,  35.38,  35.75,  35.24,  36.14,  32.79,
    34.08,  30.54,  21.74,  15.74,  20.51,  18.89,  18.19,  20.52,  23.30,

    25.53,  25.26,  27.14,  27.91,  26.80,  26.36,  26.22,  26.88,  33.59,  30.76,  32.59,  29.84,  25.00,
    26.08,  26.21,  26.12,  25.38,  23.72,  25.27,  24.72,  20.68,  18.85,  18.13,  15.94,  15.16,  17.25,
    19.02,  19.74,  22.46,  22.22,  23.42,  24.44,  26.34,  28.06,  29.09,  26.89,  27.49,  28.08,  29.28,
    24.85,  22.16,  24.86,  22.72,  23.71,  25.82,  24.61,  27.19,  27.01,  26.05,  24.88,  25.40,  24.67,
    25.64,  27.02,  26.40,  24.70,  26.43,  25.58,  25.96,  29.53,  30.77,

    84.61,  92.91,  99.80,  121.19, 122.04, 131.76, 138.48, 153.47, 189.95, 182.22, 198.08, 135.36, 125.02,
    143.50, 173.95, 188.75, 167.44, 158.95, 169.53, 113.66, 107.59, 92.67,  85.35,  90.13,  89.31,  105.12,
    125.83, 135.81, 142.43, 163.39, 168.21, 185.35, 188.50, 199.91, 210.73, 192.06, 204.62, 235.00, 261.09,
    256.88, 251.53, 257.25, 243.10, 283.75, 300.98, 311.15, 322.56, 339.32, 353.21, 348.51, 350.13, 347.83,
    335.67, 390.48, 384.83, 381.32, 404.78, 382.20, 405.00, 456.48, 493.17,

    43.45,  44.83,  46.12,  47.31,  45.39,  45.41,  48.42,  49.36,  46.53,  47.91,  46.48,  42.38,  45.24,
    46.88,  45.64,  46.50,  43.09,  47.62,  49.16,  50.66,  45.53,  39.22,  39.91,  39.77,  35.74,  33.64,
    38.67,  40.03,  43.10,  43.96,  43.93,  45.93,  48.09,  49.41,  49.99,  51.14,  49.36,  50.57,  52.00,
    50.93,  47.95,  47.71,  47.74,  48.53,  48.29,  47.70,  49.49,  48.73,  52.29,  52.70,  55.91,  56.04,
    54.40,  54.85,  53.84,  52.72,  53.73,  49.16,  50.57,  49.52,  50.21,

    28.97,  29.34,  30.59,  31.18,  32.00,  32.40,  32.49,  34.84,  34.63,  32.22,  31.46,  30.01,  28.38,
    31.69,  28.00,  26.31,  23.11,  24.50,  24.33,  22.38,  17.12,  15.07,  14.49,  10.85,  7.84,   9.31,
    11.65,  12.41,  10.88,  12.44,  12.90,  15.33,  13.32,  14.96,  14.22,  15.11,  15.19,  17.21,  17.84,
    15.46,  13.72,  15.34,  13.78,  15.58,  15.36,  15.18,  17.67,  19.46,  20.35,  19.51,  19.90,  19.11,
    18.50,  17.57,  16.00,  15.07,  16.55,  15.76,  17.91,  18.71,  19.13,

    44.68,  44.85,  44.74,  45.07,  43.45,  42.14,  45.04,  45.24,  43.45,  41.52,  37.67,  40.31,  36.28,
    35.19,  34.85,  31.57,  22.59,  31.14,  29.48,  33.79,  23.34,  15.69,  13.90,  6.50,   3.90,   6.75,
    8.84,   11.16,  13.08,  14.66,  17.43,  16.78,  14.46,  15.72,  14.94,  15.06,  16.53,  17.72,  17.70,
    15.63,  14.28,  13.95,  12.38,  13.03,  11.39,  10.89,  13.28,  13.66,  14.22,  13.28,  12.23,  11.70,
    10.92,  9.68,   8.15,   6.11,   6.82,   5.44,   5.56,   7.13,   8.18,

    19.57,  19.80,  20.74,  21.78,  20.26,  18.63,  19.92,  19.59,  19.73,  19.29,  18.45,  18.96,  18.34,
    17.23,  16.55,  16.19,  14.61,  15.61,  16.25,  15.68,  15.06,  14.21,  15.32,  12.61,  10.88,  12.04,
    11.81,  13.58,  13.41,  14.24,  15.08,  14.94,  15.37,  16.56,  16.58,  17.01,  16.15,  15.78,  15.38,
    14.16,  13.26,  13.95,  14.96,  16.14,  16.38,  15.48,  16.63,  17.30,  18.47,  19.49,  20.13,  20.79,
    19.96,  18.65,  18.60,  17.32,  18.87,  19.86,  21.42,  21.18,  21.14,

    64.20,  67.58,  71.10,  74.81,  75.45,  76.58,  77.42,  83.59,  83.08,  80.83,  84.94,  77.70,  79.23,
    77.01,  84.74,  81.18,  80.60,  73.56,  73.55,  71.39,  68.14,  74.10,  73.81,  70.71,  63.09,  63.28,
    61.95,  64.83,  65.35,  65.80,  65.03,  64.53,  67.40,  71.01,  64.50,  60.95,  61.89,  63.77,  64.52,
    57.96,  54.71,  57.21,  57.07,  59.65,  64.19,  67.58,  71.04,  78.38,  83.53,  82.16,  85.92,  81.98,
    79.93,  78.37,  73.18,  71.81,  77.20,  80.00,  84.30,  83.28,  84.88]

num_prices      = 61
num_assests     = 10
riskFreeRateFloat = 0.001
targetReturnFloat = 0.02
# Outputs - declaring them as empty lists, pybind appears only to feed out lists from a pointer referenced lower level feed, not floats
GMVPWeightsList = [] # this will be a list of values, attempting to make clear by appending 'List' to the name
GMVPVariance = [] # this will be a list containing just one float
GMVPReturn = []
EffWeightsList = []
EffVariance = []
EffReturn = []
TanWeightsList = []
TanVariance = []
TanReturn = []
TanSharpe = []
EffTanWeightsList = [] 
EffTanVariance = []
EffTanReturn = []

# Identify which cards are installed and choose the first available u200 card, as defined in deviceList above
print("Found these {0} device(s):".format(len(deviceList)))
for x in deviceList:
    print(x.getName())
print("Choosing the first suitable card\n")
chosenDevice = deviceList[0]

# Selecting and loading into FPGA on chosen card the financial model to be used
portfolio_optimisation = portfolio_optimisation(args.xclbin_filename)
portfolio_optimisation.claimDevice(chosenDevice)

#Feed in the data and request the result
print("\nRunning...")
result = portfolio_optimisation.run(testDataList, num_prices, num_assests, riskFreeRateFloat, targetReturnFloat, GMVPWeightsList, GMVPVariance, GMVPReturn, EffWeightsList, EffVariance, EffReturn, TanWeightsList, TanVariance, TanReturn, TanSharpe, EffTanWeightsList, EffTanVariance, EffTanReturn)
print("Done")

# Display results in similar format to the C++ test
print("Global Minimum Variance Portfolio")
for i in range(len(GMVPWeightsList)):
    print("GMVP w[",i,"] = ",GMVPWeightsList[i])
    i += 1
print("GMVP Expected return = ",GMVPReturn[0])
print("GMVP Variance = ",GMVPVariance[0])
print("Efficient Portfolio with target return of ",targetReturnFloat)
for i in range(len(EffWeightsList)):
    print("Eff w[",i,"] = ",EffWeightsList[i])
    i += 1
print("Eff Expected return = ",EffReturn[0])
print("Eff Variance = ",EffVariance[0])
print("Tangency Portfolio for risk free rate of ",riskFreeRateFloat)  
for i in range(len(TanWeightsList)):
    print("Tan w[",i,"] = ",TanWeightsList[i])
    i += 1
print("Tan Expected return = ",TanReturn[0])
print("Tan Variance = ",TanVariance[0])
print("Tan Sharpe Ratio = ",TanSharpe[0])
print("Tangency Portfolio for risk free rate of ",riskFreeRateFloat,"and target return ",targetReturnFloat)  
for i in range(len(EffTanWeightsList)):
    print("Eff Tan w[",i,"] = ",EffTanWeightsList[i])
    i += 1

proportionriskfree = (1.0 - (sum(float(i) for i in EffTanWeightsList))) # Matching C++ test this output is a subtraction from 1 of the sum of EffTanWeightsList
print("Eff Tan proportion in risk free = ",proportionriskfree)

#Relinquish ownership of the card
portfolio_optimisation.releaseDevice()

sys.exit(0)
