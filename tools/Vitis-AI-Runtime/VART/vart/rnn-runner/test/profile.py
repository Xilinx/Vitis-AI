#!/usr/bin/python

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
import os
import glob
import argparse
import statistics as stat

KEYS = ["INSTR_COMPUTE", "INSTR_UPLOAD", "REG_COMPUTE", "INPUT_UPLOAD", "XRT_RUN", "OUTPUT_DOWNLOAD", "EXECUTE_ASYNC"]

class ProfData:
    def __init__(self, device, model, batch, lats):
        self.device = device
        self.model = model
        self.batch = batch
        self.lats = lats

    def mean(self, key):
        return stat.mean(self.lats[key])

    def stddev(self, key):
        return stat.pstdev(self.lats[key])

def parse_file_name(logfile):
    ''' Parse a filename of the form sth_device_model_batch.lat
        and return (device, model, batch)
    '''
    fname = os.path.basename(logfile)
    f = os.path.splitext(fname)[0]
    tokens = f.split('_')
    if len(tokens) > 3:
        return (tokens[-3], tokens[-2], tokens[-1])
    else:
        return ("", "", "")

def parse_file(logfile):
    ''' Parse a file to collect item-specific latencies '''
    assert(os.path.exists(logfile))
    lats = dict((key, []) for key in KEYS)
    with open(logfile, "r") as f:
        for line in f:
            for key, _ in lats.items():
                if key in line:
                    lat_str = line.split()[-1].rstrip("us")
                    lat = float(lat_str)
                    lats[key].append(lat)
                    break
    return lats

def generate_prof_data(device, model, batch, logfile):
    lats = parse_file(logfile)
    return ProfData(device, model, batch, lats)

def generate_prof_datas(logdir, ext="lat"):
    ifiles = glob.glob(logdir + "/*." + ext)
    profs = []
    for f in ifiles:
        print(f)
        device, model, batch = parse_file_name(f)
        prof = generate_prof_data(device, model, batch, f)
        profs.append(prof)
    return profs

def save_as_csv(pdatas, outfile):
    row_headers = ["Model", "Device", "Batch"] + KEYS
    nrows = len(row_headers)
    data = [[item] for item in row_headers]
    for pdata in pdatas:
        data[0].append(pdata.model)
        data[1].append(pdata.device)
        data[2].append(pdata.batch)
        for i, key in enumerate(KEYS, 3):
            mean = round(pdata.mean(key))
            stddev = round(pdata.stddev(key))
            data[i].append(u"{} \u00b1 {}".format(mean, stddev))
    with open(outfile, 'w') as f:
        for d in data:
            f.write(','.join(d) + '\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", help="Directory containing all log files")
    parser.add_argument("--output_file", help="Path of csv file to save the result")
    args = parser.parse_args()
    pdatas = generate_prof_datas(args.log_dir)
    save_as_csv(pdatas, args.output_file)



