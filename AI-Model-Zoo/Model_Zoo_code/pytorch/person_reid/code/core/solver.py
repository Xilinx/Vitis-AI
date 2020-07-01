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


# encoding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging
import time

import numpy as np
import torch
import torch.nn.functional as F

class Solver(object):
    def __init__(self, opt, net):
        self.opt = opt
        self.net = net

    def test_func(self, test_data, num_query):
        feat, person, camera = list(), list(), list()
        for inputs in test_data:
            data, pids, camids = inputs
            with torch.no_grad():
                outputs = F.normalize(self.net(data)).cpu()
            feat.append(outputs)
            person.extend(pids.numpy())
            camera.extend(camids.numpy())
        feat = torch.cat(feat, 0)
        qf = feat[:num_query]
        q_pids = np.asarray(person[:num_query])
        q_camids = np.asarray(camera[:num_query])
        gf = feat[num_query:]
        g_pids = np.asarray(person[num_query:])
        g_camids = np.asarray(camera[num_query:])

        logging.info("Extracted features for query set, obtained {}-by-{} matrix".format(
            qf.shape[0], qf.shape[1]))
        logging.info("Extracted features for gallery set, obtained {}-by-{} matrix".format(
            gf.shape[0], gf.shape[1]))

        logging.info("Computing distance matrix")

        #qf = qf.numpy()
        #gf = gf.numpy()
        #from scipy.spatial.distance import cdist
        #distmat = cdist(qf, gf)
        m, n = qf.shape[0], gf.shape[0]
        distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                  torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        distmat.addmm_(1, -2, qf, gf.t())
        distmat = distmat.cpu().numpy()

        logging.info("Computing CMC and mAP")
        cmc, mAP = self.eval_func(distmat, q_pids, g_pids, q_camids, g_camids)

        print("Results ----------")
        print("mAP: {:.1%}".format(mAP))
        print("CMC curve")
        for r in [1, 5, 10]:
            print("Rank-{:<3}: {:.1%}".format(r, cmc[r - 1]))
        print("------------------")
        return cmc[0]

    @staticmethod
    def eval_func(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=50):
        """Evaluation with market1501 metric
            Key: for each query identity, its gallery images from the same camera view are discarded.
            """

        num_q, num_g = distmat.shape
        if num_g < max_rank:
            max_rank = num_g
            print("Note: number of gallery samples is quite small, got {}".format(num_g))
        indices = np.argsort(distmat, axis=1)
        matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

        # compute cmc curve for each query
        all_cmc = []
        all_AP = []
        num_valid_q = 0.  # number of valid query
        for q_idx in range(num_q):
            # get query pid and camid
            q_pid = q_pids[q_idx]
            q_camid = q_camids[q_idx]

            # remove gallery samples that have the same pid and camid with query
            order = indices[q_idx]

            remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
            keep = np.invert(remove)
            orig_cmc = matches[q_idx][keep]

            # compute cmc curve
            # binary vector, positions with value 1 are correct matches
            if not np.any(orig_cmc):
                # this condition is true when query identity does not appear in gallery
                continue

            cmc = orig_cmc.cumsum()
            cmc[cmc > 1] = 1

            all_cmc.append(cmc[:max_rank])
            num_valid_q += 1.

            # compute average precision
            # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
            num_rel = orig_cmc.sum()
            tmp_cmc = orig_cmc.cumsum()
            tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
            tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
            AP = tmp_cmc.sum() / num_rel
            all_AP.append(AP)

        assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

        all_cmc = np.asarray(all_cmc).astype(np.float32)
        all_cmc = all_cmc.sum(0) / num_valid_q
        mAP = np.mean(all_AP)

        return all_cmc, mAP
