.. 
   Copyright 2019 Xilinx, Inc.
  
   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at
  
       http://www.apache.org/licenses/LICENSE-2.0
  
   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

.. meta::
   :keywords: finance model
   :description: A series of classes are provided that represent the various financial models that are supported. These classes provide all the methods that are required to run that financial model on a given HW device.
   :xlnxdocumentclass: Document
   :xlnxdocumenttype: Tutorials

***************
Running a Model
***************

A series of classes are provided that represent the various financial models that are supported.
These classes provide all the methods that are required to run that financial model on a given HW device.

Basic Flow
**********

To run a model there are 4 basic steps:
	* Instantiate an instance of the relevant class
	* Claim the device for use by that model instance
	* Run the model (as many times as required)
	* Release the device so it may be used by other objects.


Example
*******
.. code-block:: c++
	:linenos:

	#include <vector>
	#include "xf_fintech_api.hpp"

	using namespace xf::fintech;

	int retval = XLNX_OK;
	int i;

	// Instantiate a Monte-Carlo European object...
	MCEuropean mcEuropean;

	// Some input data...
	OptionType  optionType   = Put;
	double stockPrice        = 36.0;
	double strikePrice       = 40.0;
	double riskFreeRate      = 0.06;
	double dividendYield     = 0.0;
	double volatility        = 0.20;
	double timeToMaturity    = 1.0;	/* in years */
	double requiredTolerance = 0.02;
	
	double optionPrice;	// output

	// Claim the device for use by our object...
	retval = mcEuropean.claimDevice(pChosenDevice);

	if(retval == XLNX_OK)
	{
		for(i=0; i<100; i++)
		{
			//Run the model...
			retval = mcEuropean.run(optionType, stockPrice, strikePrice, riskFreeRate, dividendYield,volatility, timeToMaturity, requiredTolerance, &optionPrice);

			if(retval != XLNX_OK)
			{
				//TODO report error
				break; // out of loop 
			}
		}
	}

	//Release the device...
	retval = mcEuropean.releaseDevice();




In the above example *pChosenDevice* is a pointer to a Device object.  This acquired by enumerating available devices (see :ref:`Device Enumeration <device_enumeration>`)
  

Claiming the device
*******************

When an object claims the device, several operations occur:
	* The relevant XCLBIN (bitstream) that implements that financial model is downloaded to the FPGA if necessary.
	  This can take several seconds (depending on the size of the XCLBIN file).
	  NOTE - If the new XCLBIN is the same as the XCLBIN currently programmed on the device, this step is internally skipped.
	* The device is "locked" for use by the specified object.

Because the user may instantiate several different model objects in software, but the FPGA can only be programmed with a single XCLBIN file at a time (and that XCLBIN contains the 
implementation of possibly only a single model), it is necessary to "restrict" execution to only objects that match that XCLBIN.

To switch between several different models the user must:
	* "Claim" the device for the **first** model
	* Run that model (as many times as required)
	* Release the device
	* "Claim" the device for the **second** model
	* Run that model (as many times as required)
	* Release the device


Example of using multiple models
*********************************
.. code-block:: c++
	:linenos:

	#include <vector>
	#include "xf_fintech_api.hpp"

	using namespace xf::fintech;

	int retval = XLNX_OK;
	int i;

	// Instantiate both a Monte-Carlo European and a Monte-Carlo American object...
	MCEuropean mcEuropean;
	MCAmerican mcAmerican;

	// Some input data...
	OptionType  optionType   = Put;
	double stockPrice        = 36.0;
	double strikePrice       = 40.0;
	double riskFreeRate      = 0.06;
	double dividendYield     = 0.0;
	double volatility        = 0.20;
	double timeToMaturity    = 1.0;	/* in years */
	double requiredTolerance = 0.02;
	
	double optionPrice;	// output

	// Claim the device for use by our FIRST model object...
	retval = mcEuropean.claimDevice(pChosenDevice);

	if(retval == XLNX_OK)
	{
		for(i=0; i<100; i++)
		{
			//Run the model...
			retval = mcEuropean.run(optionType, stockPrice, strikePrice, riskFreeRate, dividendYield,volatility, timeToMaturity, requiredTolerance, &optionPrice);

			if(retval != XLNX_OK)
			{
				//TODO report error
				break; // out of loop 
			}
		}
	}

	//Release the device...
	retval = mcEuropean.releaseDevice();

	//////////////////////////////////////////////////////////////////////////////////

	// Claim the device for use by our SECOND model object...
	retval = mcAmerican.claimDevice(pChosenDevice);

	if(retval == XLNX_OK)
	{
		for(i=0; i<100; i++)
		{
			//Run the model...
			retval = mcAmerican.run(optionType, stockPrice, strikePrice, riskFreeRate, dividendYield,volatility, timeToMaturity, requiredTolerance, &optionPrice);

			if(retval != XLNX_OK)
			{
				//TODO report error
				break; // out of loop 
			}
		}
	}

	//Release the device...
	retval = mcAmerican.releaseDevice();


Notes
*****

	* A device may only be claimed by a single model object at a time.
	* A model object may only claim a single device at a time.


