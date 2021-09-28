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
   :keywords: device, enumeration, DeviceManager, getDeviceList, 
   :description: In order to run any of the models, they must be associated with a particular hardware device. The DeviceManager utility class is used to discover the devices available on a given system. 
   :xlnxdocumentclass: Document
   :xlnxdocumenttype: Tutorials


.. _device_enumeration:

********************************
Device Enumeration
********************************
   
In order to run any of the models, they must be associated with a particular hardware device. The DeviceManager utility class made available and is used to discover the devices available on a given system. 


The following is an example to show how to enumerate ALL available Xilinx devices on a system:

.. code-block:: c++
	:linenos:

	#include <vector>
	#include "xf_fintech_api.hpp"

	using namespace xf::fintech;



	std::vector<Device*> deviceList;
	Device* pChosenDevice;



	deviceList = DeviceManager::getDeviceList();

	if (deviceList.size() == 0)
	{
		printf("No matching devices found\n");
		exit(0);
	}

	printf("Found %zu matching devices\n", deviceList.size());


	//we'll just pick the first device in the list...
	pChosenDevice = deviceList[0];





This next example shows how to enumerate specific types of Xilinx devices based on their device name.  This is useful if you have multiple types of cards installed on your system.


The **getDeviceList** method takes a string parameter that is used to perform a substring comparison on the full names of each of the available devices.


NOTE - the string parameter is case-sensitive:

.. code-block:: c++
	:linenos:

	#include <vector>
	#include "xf_fintech_api.hpp"

	using namespace xf::fintech;



	std::vector<Device*> deviceList;
	Device* pChosenDevice;


	// Get a list of U250s available on the system 
	// (just because our current bitstreams are built for U250s)
	deviceList = DeviceManager::getDeviceList("u250");

	if (deviceList.size() == 0)
	{
		printf("No matching devices found\n");
		exit(0);
	}

	printf("Found %zu matching devices\n", deviceList.size());


	//we'll just pick the first device in the list...
	pChosenDevice = deviceList[0];




