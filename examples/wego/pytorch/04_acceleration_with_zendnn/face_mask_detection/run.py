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

import os
import argparse
import threading
import time
from tqdm import tqdm
import torch
import wego_torch

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', help='The path of quantized model', type=str, default='/tmp/wego_example_recipes/zendnn')
parser.add_argument('--threads', help='The threads number for running the applications', type=int, default=8)
parser.add_argument('--enable_zen', help='Enable zendnn or not', default=False, action="store_true")

args = parser.parse_args()

def generate_inputs_data(batch, inputs_shapes):
    inputs_data = []
    for i, shape in enumerate(inputs_shapes):
        data = torch.randint(1, 3, shape).to(torch.float)
        batch_data = torch.cat([data for i in range(batch)], 0)
        print("[Info] input shape for input-%d is %s\n" % (i, batch_data.shape))
        inputs_data.append(batch_data)
    return inputs_data

def run_thread(model, inputs_data, cnt, n_of_group, step):
    for count in range(cnt, n_of_group, step):
        model(*inputs_data)
    
def run_throughput(model, inputs_data, n_thread=1, n_of_group=1200):
    threads = []
    for i in range(n_thread):
        tr = threading.Thread(target=run_thread, args=(model, inputs_data, i, n_of_group, n_thread))
        threads.append(tr)

    start_t = time.perf_counter()
    for x in threads:
        x.start()
    for x in threads:
        x.join()
    end_t = time.perf_counter()
    return end_t - start_t

def run_perf(args, model, inputs_data, batch):
    n_of_group = 100
    r_n = 5
    t = 0.0
    print("[Info] running with total %d rounds, each of which has %d images" % (r_n, n_of_group * batch))
    for i in tqdm(range(r_n)):
        t_ = run_throughput(model, inputs_data, args.threads, n_of_group)
        t += t_
    print("===================== Perf Result =====================")
    print("[Total Images] %d" % (r_n * n_of_group * batch))
    print("[Total Time]   %0.6fs" % float(t))
    print("[FPS]          %0.2f" % (float(n_of_group * batch) / (t / r_n)))

def main(): 
    # loading original quantized torchscript module
    model_path = args.model_path
    mod = torch.jit.load(model_path)

    enable_zen = args.enable_zen
    print(f"[Info] ZenDNN is {'enabled' if enable_zen else 'disabled'}!")

    # Create wego module
    wego_mod = wego_torch.compile(mod, wego_torch.CompileOptions(
        inputs_meta = [
        wego_torch.InputMeta(torch.float, [1, 3, 512, 512])
        ],
        optimize_options = wego_torch.OptimizeOptions(zendnn_enable=enable_zen),
    ))
    wego_mod.eval()
     
    target_info = wego_torch.get_target_info()
    print("[Info] target_info: %s" % str(target_info))
    batch = target_info.batch
    inputs_data = generate_inputs_data(batch, [(1, 3, 512, 512)])

    print("[Info] warming up...")
    wego_mod(*inputs_data)

    print(f"[Info] start running throughput perf test with ZenDNN {'enabled' if enable_zen else 'disabled'}")
    run_perf(args, wego_mod, inputs_data, batch)  
    
if __name__ == '__main__':
    main()
