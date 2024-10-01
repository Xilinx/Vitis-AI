#!/usr/bin/python3

# Copyright 2022-2023 Advanced Micro Devices Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
import os
import xir
import dpu_mc_parser
import logging
import binascii
import argparse
import time
import csv

PARSE_TIME_ERR = 30
LOG_PATH = "./test_dpu_mc.log"
CSV_PATH = "./test_dpu_mc.csv"


def find_xmodel(dir):
    suffix = ".xmodel"
    ret = []
    for root, dirs, files in os.walk(dir):
        s = [x for x in files if x.endswith(suffix)]
        for x in s:
            ret.append(os.path.join(root, x))

    return ret


def xmodel_to_subgraphs(xmodel):
    graph = xir.Graph.deserialize(xmodel)
    root = graph.get_root_subgraph()

    # add subgraphs of depth=1
    ret = list(root.toposort_child_subgraph())

    # add subgraphs of depth=2
    for dp1_subgraph in root.toposort_child_subgraph():
        for dp2_subgraph in dp1_subgraph.toposort_child_subgraph():
            ret.append(dp2_subgraph)

    return root.toposort_child_subgraph()


def subgraph_to_mc_str(subg):
    name = subg.get_name()

    if subg.has_attr("mc_code"):
        mc = subg.get_attr("mc_code")
        mc_str = binascii.b2a_hex(mc)
        return mc_str

    return None


def process_mc_str(dpu_name, mc_str):
    mc_data = dpu_mc_parser.process_mc(dpu_name, mc_str, True)
    load_img_size = mc_data[0]
    load_para_size = mc_data[1]
    save_size = mc_data[2]
    workload = mc_data[3]

    return load_img_size, load_para_size, save_size, workload


if __name__ == "__main__":
    # process logging
    logging.basicConfig(level=logging.INFO)
    os.system("echo > %s" % LOG_PATH)
    fhlr = logging.FileHandler(LOG_PATH)
    logging.getLogger().addHandler(fhlr)

    # process args
    cmd_parser = argparse.ArgumentParser(
        usage="Usage: ./test_dpu_mc.py <xmodel_dir_path> -d <depth>\ne.g: ./writer/parser/test_dpu_mc.py /usr/share/vitis_ai_library/models/ -d 1")
    cmd_parser.add_argument('-d', '--depth', default=1,
                            choices=[1, 2], type=int, help="xmodel parse depth, default: 1")
    cmd_parser.add_argument("xmodel_dir", nargs=1, help="path to xmodel dir")
    args = cmd_parser.parse_args()

    xm_dir = args.xmodel_dir[0]
    xmodel_parse_depth = args.depth

    if not os.path.isdir(xm_dir):
        raise RuntimeError("Invalid directory")

    xmodels = find_xmodel(xm_dir)
    n_models = len(xmodels)

    logging.info("%s%s%s" % (50 * "=", "[START]", 50 * "="))
    csv_header = ['xmodel', 'dpu_name', 'subgraph_name', 'subgraph_depth',
                  'load_img_size', "load_para_size", "save_size", "workload", "mc_len"]
    csv_f = open(CSV_PATH, 'w', encoding='utf-8')
    csv_writer = csv.writer(csv_f)
    csv_writer.writerow(csv_header)

    for m in xmodels:
        subgs = xmodel_to_subgraphs(m)
        xmodel_name = m.split('/')[-1]
        logging.info("Processing xmodel [%d/%d]: %s, # of subgraphs: %d" %
                     (xmodels.index(m) + 1, n_models, m, len(subgs)))
        logging_prefix = 4 * " "

        for subg in subgs:
            t_subgraph_start = time.time()
            if (subg.has_attr("device")):
                device = subg.get_attr("device")
                if device != "DPU":
                    continue

            if (subg.has_attr("dpu_name")):
                dpu_name = subg.get_attr("dpu_name")
            else:
                logging.error("Cannot find <dpu_name> in xmodel")

            # process depth=1 subgraphs
            mc_str = subgraph_to_mc_str(subg)
            if mc_str != None:
                logging.info(logging_prefix + "Processing subgraph: %s, dpu_name: %s" %
                             (subg.get_name(), dpu_name))
                load_img_size, load_para_size, save_size, workload = process_mc_str(
                    dpu_name, mc_str)
                if workload == 0:
                    """Try to get workload from xmodel"""
                    if subg.has_attr("workload"):
                        workload = subg.get_attr("workload")
                logging.info(logging_prefix + "Result: LoadImgSize: {:,}, LoadParaSize: {:,}, SaveSize: {:,}, Workload: {:,}".format
                             (load_img_size, load_para_size, save_size, workload))
                t_subgraph_dur = (time.time() - t_subgraph_start)
                if t_subgraph_dur < PARSE_TIME_ERR:
                    logging.info(
                        logging_prefix + "Parsing time consuming: {:.3f} s, mc_size: {:,}".format(t_subgraph_dur, len(mc_str)))
                else:
                    logging.error(
                        logging_prefix + "ERROR: Parsing time consuming: {:.3f} s, mc_size: {:,}".format(t_subgraph_dur, len(mc_str)))
                logging.info(logging_prefix + 80 * "-")
                #csv_header = ['xmodel', 'dpu_name', 'subgraph_name', 'subgraph_depth', 'load_img_size', "load_para_size", "save_size", "workload", "mc_len"]
                csv_writer.writerow([xmodel_name, dpu_name, subg.get_name(
                ), 1, load_img_size, load_para_size, save_size, workload, len(mc_str)])

            # process depth=1 subgraphs
            if xmodel_parse_depth == 1:
                continue
            for dp2_subg in subg.toposort_child_subgraph():
                mc_str = subgraph_to_mc_str(dp2_subg)
                if mc_str != None:
                    logging.info(logging_prefix + "Processing subgraph of depth=2: %s, dpu_name: %s" %
                                 (dp2_subg.get_name(), subg.get_name(), ))
                    load_img_size, load_para_size, save_size, workload = process_mc_str(
                        dpu_name, mc_str)

                    if workload == 0:
                        """Try to get workload from xmodel"""
                        if dp2_subg.has_attr("workload"):
                            workload = dp2_subg.get_attr("workload")
                    logging.info(logging_prefix + "Result: LoadImgSize: {:,}, LoadParaSize: {:,}, SaveSize: {:,}, Workload: {:,}".format
                                 (load_img_size, load_para_size, save_size, workload))
                    logging.info(logging_prefix + 80 * "-")
                    csv_writer.writerow([xmodel_name, dpu_name, dp2_subg.get_name(
                    ), 2, load_img_size, load_para_size, save_size, workload, len(mc_str)])
        logging.info(80 * "-")

    logging.info("%s%s%s" % (50 * "=", "[END]", 50 * "="))
    logging.warning("Log File: [%s]" % os.path.abspath(LOG_PATH))
    logging.warning("CSV File: [%s]" % os.path.abspath(CSV_PATH))
    csv_f.close()
