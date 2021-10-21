# Copyright 2019 Xilinx, Inc.
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

 # Copyright 2019 Xilinx, Inc.
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
 
import os
import subprocess
import argparse

def runApp(hostExe, xclbin, sigPath, vecPath, mtxName, numIters, tolerance, deviceId):
    res = subprocess.run([hostExe, xclbin, numIters, tolerance, sigPath, vecPath, mtxName, deviceId], universal_newlines=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return res.returncode, res.stdout, res.stderr

def process(hostExePath, xclbinPath, sigPath, vecPath, numIters, tolerance, deviceId):
    csvFile = open("./cgsolver_perf.csv", "w")
    csvFile.write("matrix name, original rows/cols, original NNZs, padded rows/cols, padded NNZs, padding ratio, num of iters, total run time[sec], time[ms]/run, num_mismatches\n")
    hostExe = os.path.join(hostExePath, "host.exe")
    xclbin = os.path.join(xclbinPath, "cgSolver.xclbin")
    sigPath.strip()
    sigPath.rstrip("/")
    vecPath.strip()
    vecPath.rstrip("/")
    mtxNames = os.listdir(sigPath)

    numErrs = 0;
    for mtxName in mtxNames:
        print("Running test with {}".format(mtxName))
        exitCode, outTxt, errTxt = runApp(hostExe, xclbin, sigPath, vecPath, mtxName, str(numIters), tolerance, str(deviceId))
        for item in outTxt.split("\n"):
            if "DATA_CSV:," in item:
                item = item.lstrip("DATA_CSV:,")
                csvFile.write(item+"\n")
        if (exitCode == 0):
            print("Test pass!")
        else:
            print("Stdout: "+outTxt)
            print("Stderr: "+errTxt)
            numErrs += 1
    csvFile.close()
    if (numErrs == 0):
        print ("All tests pass!")
        print ("Please find the benchmark results in cgsolver_perf.csv.")
    else:
        print("Some tests failed! Total {} failed tests!".format(numErrs))

def main(args):
    if (args.usage):
        print('Usage example:')
        print('python run_test.py --host_path ./build_dir.hw.xilinx_u280_xdma_201920_3 --xclbin_path ./build_dir.hw.xilinx_u280_xdma_201920_3 --sig_path ./sig_dat --vec_path ./vec_dat --num_iters 5000 --tolerance 1e-12 --device_id 0')
    else:
        process(args.host_path, args.xclbin_path, args.sig_path, args.vec_path, args.num_iters, args.tolerance, args.device_id)
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='run CG Solver benchmark test')
    parser.add_argument('--usage',action='store_true',help='print usage example')
    parser.add_argument('--host_path',type=str,default='./build_dir.hw.xilinx_u280_xdma_201920_3',help='directory for host executable')
    parser.add_argument('--xclbin_path',type=str,default='./build_dir.hw.xilinx_u280_xdma_201920_3',help='directory for xclbin file')
    parser.add_argument('--sig_path',type=str,default='./sig_dat',help='directory for matrix signature data')
    parser.add_argument('--vec_path',type=str,default='./vec_dat',help='directory for vector data')
    parser.add_argument('--num_iters',type=int,default=5000,help='number of times that SPMV operations is carried out in hardware')
    parser.add_argument('--tolerance',type=str,default='1e-12',help='tolerance for convergency')
    parser.add_argument('--device_id',type=int,default=0,help='device ID for the U280 FPGA card')
    args = parser.parse_args()
    main(args)
  
