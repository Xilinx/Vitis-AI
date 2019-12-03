# Copyright 2019 Xilinx Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import sys
from xrt_binding import *

# main
def main():
	deviceInfo = xclDeviceInfo2()
	handle = xclDeviceHandle
	for i in range(0, xclProbe()):
		handle = xclOpen(i, None, xclVerbosityLevel.XCL_QUIET)
		if xclGetDeviceInfo2(handle, ctypes.byref(deviceInfo)):
			print("Error 2")
			return -1
		else:
			print deviceInfo.mName
	return 0

if __name__== "__main__" :
	main()
