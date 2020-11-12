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

#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.

import imp
import os
import time
import pytorch_nndct
from pytorch_nndct.apis import torch_quantizer, dump_xmodel
from nndct_shared.utils import  set_option_value
import __init__ as booger
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from tasks.semantic.modules.segmentator import *
from tasks.semantic.postproc.KNN import KNN
from torch import nn

#set_option_value("nndct_quant_off", True)
 


class User():
    def __init__(self, ARCH, DATA, datadir, logdir, modeldir,modelname,split,quant_mode,dump,device):
        # parameters
        self.ARCH = ARCH
        self.DATA = DATA
        self.datadir = datadir
        self.logdir = logdir
        self.modeldir = modeldir
        self.modelname = modelname
        self.split = split
        self.quant_mode = quant_mode
        self.dump = dump
        self.device = device

        # get the data
        parserModule = imp.load_source("parserModule",
                                   booger.TRAIN_PATH + '/tasks/semantic/dataset/' +
                                   self.DATA["name"] + '/parser.py')
        self.parser = parserModule.Parser(root=self.datadir,
                                          train_sequences=self.DATA["split"]["train"],
                                          valid_sequences=self.DATA["split"]["valid"],
                                          test_sequences=self.DATA["split"]["test"],
                                          labels=self.DATA["labels"],
                                          color_map=self.DATA["color_map"],
                                          learning_map=self.DATA["learning_map"],
                                          learning_map_inv=self.DATA["learning_map_inv"],
                                          sensor=self.ARCH["dataset"]["sensor"],
                                          max_points=self.ARCH["dataset"]["max_points"],
                                          batch_size=1,
                                          workers=1,
                                          gt=True,
                                          shuffle_train=False)

        # concatenate the encoder and the head
        if self.modelname in ('salsanet','salsanext'):
            with torch.no_grad():

                self.model = SalsaNet(self.ARCH,
                                      self.parser.get_n_classes(),
                                      self.modeldir)
                #self.model = nn.DataParallel(self.model)
                torch.nn.Module.dump_patches = True
                w_dict = torch.load(modeldir + "/SalsaNet",
                                    map_location=lambda storage, loc: storage)


                #load the weights
                model_dict = self.model.state_dict()
                new_dict = {}
                for k,v in w_dict['state_dict'].items():
                    if k[0:7] == 'module.':
                        name = k[7:]
                        if name in model_dict.keys():
                            new_dict[name] = v
                    else:
                        if k in model_dict.keys():
                            new_dict[k] = v

                model_dict.update(new_dict)
                #self.model.load_state_dict(w_dict['state_dict'], strict=True)
                self.model.load_state_dict(model_dict)
        else:
            with torch.no_grad():
                self.model = Segmentator(self.ARCH,
                                      self.parser.get_n_classes(),
                                      self.modeldir)

        # use knn post processing?
        self.post = None
        if self.ARCH["post"]["KNN"]["use"]:
            self.post = KNN(self.ARCH["post"]["KNN"]["params"],
                            self.parser.get_n_classes())

        # GPU?
        self.gpu = False
        self.model_single = self.model
        self.device = torch.device("cuda" if torch.cuda.is_available() and device=='gpu' else "cpu")
        print("Infering in device: ", self.device)
        if torch.cuda.is_available() and device == 'gpu' and torch.cuda.device_count() > 0:
            cudnn.benchmark = True
            cudnn.fastest = True
            self.gpu = True
            self.model.cuda()

    def infer(self):
        # do train set
        if self.split == None:
            self.infer_subset(loader=self.parser.get_train_set(),
                              to_orig_fn=self.parser.to_original)

            # do valid set
            self.infer_subset(loader=self.parser.get_valid_set(),
                              to_orig_fn=self.parser.to_original)
            # do test set
            self.infer_subset(loader=self.parser.get_test_set(),
                              to_orig_fn=self.parser.to_original)
        elif self.split == 'valid':
            self.infer_subset(loader=self.parser.get_valid_set(),
                              to_orig_fn=self.parser.to_original,
                              quant_mode=self.quant_mode,
                              dump=self.dump)
        elif self.split == 'train':
            self.infer_subset(loader=self.parser.get_train_set(),
                              to_orig_fn=self.parser.to_original)
        else:
            self.infer_subset(loader=self.parser.get_test_set(),
                              to_orig_fn=self.parser.to_original)

        print('Finished Infering')

        return

    def infer_subset(self, loader, to_orig_fn,quant_mode,dump):
        # switch to evaluate mode
        self.model.eval()
        if self.gpu:
            device = torch.device("cuda")
            torch.cuda.empty_cache()
        else:
            device = torch.device("cpu")
        if quant_mode == 'calib':
            input = torch.randn([1, 5,64 ,2048])
            quantizer = torch_quantizer(quant_mode, self.model, (input), device=device)
            quant_model = quantizer.quant_model
        if quant_mode == 'test':
            input = torch.randn([1, 5,64 ,2048])
            quantizer = torch_quantizer(quant_mode, self.model, (input), device=device)
            quant_model = quantizer.quant_model
        #quant_model.eval()


        # empty the cache to infer in high res
        with torch.no_grad():
            end = time.time()

            for i, (proj_in, proj_mask, _, _, path_seq, path_name, p_x, p_y, proj_range, unproj_range, _, _, _, _,
                    npoints) in enumerate(loader):
                #first cut to rela size (batch size one allows it)
                p_x = p_x[0, :npoints]
                p_y = p_y[0, :npoints]
                proj_range = proj_range[0, :npoints]
                unproj_range = unproj_range[0, :npoints]
                path_seq = path_seq[0]
                path_name = path_name[0]

#                if self.gpu:
#                    proj_in = proj_in.cuda()
#                    p_x = p_x.cuda()
#                    p_y = p_y.cuda()
#                    if self.post:
#                        proj_range = proj_range.cuda()
#                        unproj_range = unproj_range.cuda()
#                 else:
                proj_in = proj_in.to(device)
                p_x = p_x.to(device)
                p_y = p_y.to(device)
                if self.post:
                    proj_range = proj_range.to(device)
                    unproj_range = unproj_range.to(device)

                # compute output
                if quant_mode == "calib" or quant_mode == 'test':
                    proj_output = quant_model(proj_in)
                else:
                    proj_output = self.model(proj_in)
                proj_argmax = proj_output[0].argmax(dim=0)

                if self.post:
                    # knn postproc
                    unproj_argmax = self.post(proj_range,
                                              unproj_range,
                                              proj_argmax,
                                              p_x,
                                              p_y)
                else:
                    # put in original pointcloud using indexes
                    unproj_argmax = proj_argmax[p_y, p_x]

                # measure elapsed time
                #if torch.cuda.is_available():
                #    torch.cuda.synchronize()

                print("Infered seq", path_seq, "scan", path_name,
                      "in", time.time() - end, "sec")
                end = time.time()

                # save scan
                # get the first scan in batch and project scan
                pred_np = unproj_argmax.cpu().numpy()
                pred_np = pred_np.reshape((-1)).astype(np.int32)

                # map to original label
                pred_np = to_orig_fn(pred_np)

                # save scan
                path = os.path.join(self.logdir, "sequences",
                                    path_seq, "predictions", path_name)
                pred_np.tofile(path)
                if (i==1 and dump and str(self.device)=='cpu' and os.environ["W_QUANT"]=='1'):
                    dump_xmodel(deploy_check=True)
                    break;
            if quant_mode == 'calib':
               quantizer.export_quant_config()
