import os, sys
import matplotlib.pyplot as plt
import numpy as np
import pdb
import time
import copy

def target(x,temp):
  gam = 4;
  val = gam*(x**2-1)*(x**2-1);
  retval = np.exp(-val)**(1/temp);
  return retval
# def target(x,temp):
#   #pragma HLS inline
#   result = 1/(1 + x*x);
#   return result


def plot_hist():

    fin = open('vitis_samples_out.csv','rt')
    data = fin.read()
    data = data.split('\n')
    data = [float(x) for x in data]

    minn = -2
    maxx = +2
    bins = 100

    sampleBins = np.linspace(minn,maxx,bins)

    y = target(sampleBins,1)

    #plt.hist(data, bins=sampleBins, density=True)
    count,bins = np.histogram(data, bins=sampleBins)
    width = 0.8 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    plt.plot(sampleBins, y/np.sum(y), color='red')
    plt.bar(center, count/np.sum(count), align='center', width=width, color='blue')
    plt.show()



if __name__ == "__main__":
    exit(plot_hist())
