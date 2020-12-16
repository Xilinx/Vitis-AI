# Apache License
# 
# Version 2.0, January 2004
# 
# http://www.apache.org/licenses/
# 
# TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION
# 
# 1. Definitions.
# 
# "License" shall mean the terms and conditions for use, reproduction, and distribution as defined by Sections 1 through 9 of this document.
# 
# "Licensor" shall mean the copyright owner or entity authorized by the copyright owner that is granting the License.
# 
# "Legal Entity" shall mean the union of the acting entity and all other entities that control, are controlled by, or are under common control with that entity. For the purposes of this definition, "control" means (i) the power, direct or indirect, to cause the direction or management of such entity, whether by contract or otherwise, or (ii) ownership of fifty percent (50%) or more of the outstanding shares, or (iii) beneficial ownership of such entity.
# 
# "You" (or "Your") shall mean an individual or Legal Entity exercising permissions granted by this License.
# 
# "Source" form shall mean the preferred form for making modifications, including but not limited to software source code, documentation source, and configuration files.
# 
# "Object" form shall mean any form resulting from mechanical transformation or translation of a Source form, including but not limited to compiled object code, generated documentation, and conversions to other media types.
# 
# "Work" shall mean the work of authorship, whether in Source or Object form, made available under the License, as indicated by a copyright notice that is included in or attached to the work (an example is provided in the Appendix below).
# 
# "Derivative Works" shall mean any work, whether in Source or Object form, that is based on (or derived from) the Work and for which the editorial revisions, annotations, elaborations, or other modifications represent, as a whole, an original work of authorship. For the purposes of this License, Derivative Works shall not include works that remain separable from, or merely link (or bind by name) to the interfaces of, the Work and Derivative Works thereof.
# 
# "Contribution" shall mean any work of authorship, including the original version of the Work and any modifications or additions to that Work or Derivative Works thereof, that is intentionally submitted to Licensor for inclusion in the Work by the copyright owner or by an individual or Legal Entity authorized to submit on behalf of the copyright owner. For the purposes of this definition, "submitted" means any form of electronic, verbal, or written communication sent to the Licensor or its representatives, including but not limited to communication on electronic mailing lists, source code control systems, and issue tracking systems that are managed by, or on behalf of, the Licensor for the purpose of discussing and improving the Work, but excluding communication that is conspicuously marked or otherwise designated in writing by the copyright owner as "Not a Contribution."
# 
# "Contributor" shall mean Licensor and any individual or Legal Entity on behalf of whom a Contribution has been received by Licensor and subsequently incorporated within the Work.
# 
# 2. Grant of Copyright License. Subject to the terms and conditions of this License, each Contributor hereby grants to You a perpetual, worldwide, non-exclusive, no-charge, royalty-free, irrevocable copyright license to reproduce, prepare Derivative Works of, publicly display, publicly perform, sublicense, and distribute the Work and such Derivative Works in Source or Object form.
# 
# 3. Grant of Patent License. Subject to the terms and conditions of this License, each Contributor hereby grants to You a perpetual, worldwide, non-exclusive, no-charge, royalty-free, irrevocable (except as stated in this section) patent license to make, have made, use, offer to sell, sell, import, and otherwise transfer the Work, where such license applies only to those patent claims licensable by such Contributor that are necessarily infringed by their Contribution(s) alone or by combination of their Contribution(s) with the Work to which such Contribution(s) was submitted. If You institute patent litigation against any entity (including a cross-claim or counterclaim in a lawsuit) alleging that the Work or a Contribution incorporated within the Work constitutes direct or contributory patent infringement, then any patent licenses granted to You under this License for that Work shall terminate as of the date such litigation is filed.
# 
# 4. Redistribution. You may reproduce and distribute copies of the Work or Derivative Works thereof in any medium, with or without modifications, and in Source or Object form, provided that You meet the following conditions:
# 
# You must give any other recipients of the Work or Derivative Works a copy of this License; and
# You must cause any modified files to carry prominent notices stating that You changed the files; and
# You must retain, in the Source form of any Derivative Works that You distribute, all copyright, patent, trademark, and attribution notices from the Source form of the Work, excluding those notices that do not pertain to any part of the Derivative Works; and
# If the Work includes a "NOTICE" text file as part of its distribution, then any Derivative Works that You distribute must include a readable copy of the attribution notices contained within such NOTICE file, excluding those notices that do not pertain to any part of the Derivative Works, in at least one of the following places: within a NOTICE text file distributed as part of the Derivative Works; within the Source form or documentation, if provided along with the Derivative Works; or, within a display generated by the Derivative Works, if and wherever such third-party notices normally appear. The contents of the NOTICE file are for informational purposes only and do not modify the License. You may add Your own attribution notices within Derivative Works that You distribute, alongside or as an addendum to the NOTICE text from the Work, provided that such additional attribution notices cannot be construed as modifying the License.
# 
# You may add Your own copyright statement to Your modifications and may provide additional or different license terms and conditions for use, reproduction, or distribution of Your modifications, or for any such Derivative Works as a whole, provided Your use, reproduction, and distribution of the Work otherwise complies with the conditions stated in this License.
# 5. Submission of Contributions. Unless You explicitly state otherwise, any Contribution intentionally submitted for inclusion in the Work by You to the Licensor shall be under the terms and conditions of this License, without any additional terms or conditions. Notwithstanding the above, nothing herein shall supersede or modify the terms of any separate license agreement you may have executed with Licensor regarding such Contributions.
# 
# 6. Trademarks. This License does not grant permission to use the trade names, trademarks, service marks, or product names of the Licensor, except as required for reasonable and customary use in describing the origin of the Work and reproducing the content of the NOTICE file.
# 
# 7. Disclaimer of Warranty. Unless required by applicable law or agreed to in writing, Licensor provides the Work (and each Contributor provides its Contributions) on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied, including, without limitation, any warranties or conditions of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A PARTICULAR PURPOSE. You are solely responsible for determining the appropriateness of using or redistributing the Work and assume any risks associated with Your exercise of permissions under this License.
# 
# 8. Limitation of Liability. In no event and under no legal theory, whether in tort (including negligence), contract, or otherwise, unless required by applicable law (such as deliberate and grossly negligent acts) or agreed to in writing, shall any Contributor be liable to You for damages, including any direct, indirect, special, incidental, or consequential damages of any character arising as a result of this License or out of the use or inability to use the Work (including but not limited to damages for loss of goodwill, work stoppage, computer failure or malfunction, or any and all other commercial damages or losses), even if such Contributor has been advised of the possibility of such damages.
# 
# 9. Accepting Warranty or Additional Liability. While redistributing the Work or Derivative Works thereof, You may choose to offer, and charge a fee for, acceptance of support, warranty, indemnity, or other liability obligations and/or rights consistent with this License. However, in accepting such obligations, You may act only on Your own behalf and on Your sole responsibility, not on behalf of any other Contributor, and only if You agree to indemnify, defend, and hold each Contributor harmless for any liability incurred by, or claims asserted against, such Contributor by reason of your accepting any such warranty or additional liability.
# 
# END OF TERMS AND CONDITIONS

from __future__ import print_function
try:
    import cPickle as pickle
except ImportError:
    import pickle
import cv2
import os
import numpy as np
import json
from PIL import Image

from ..logger import logger
from .imdb import IMDB
from .ds_utils import unique_boxes, filter_small_boxes
from ..config import config

class retinaface(IMDB):
    def __init__(self, image_set, root_path, data_path):
        super(retinaface, self).__init__('retinaface', image_set, root_path, data_path)
        #assert image_set=='train'

        split = image_set
        self._split = image_set
        self._image_set = image_set

        self.root_path = root_path
        self.data_path = data_path


        self._dataset_path = self.data_path
        self._imgs_path = os.path.join(self._dataset_path, image_set, 'images')
        self._fp_bbox_map = {}
        label_file = os.path.join(self._dataset_path, image_set, 'label.txt')
        name = None
        for line in open(label_file, 'r'):
          line = line.strip()
          if line.startswith('#'):
            name = line[1:].strip()
            self._fp_bbox_map[name] = []
            continue
          assert name is not None
          assert name in self._fp_bbox_map
          self._fp_bbox_map[name].append(line)
        print('origin image size', len(self._fp_bbox_map))

        #self.num_images = len(self._image_paths)
        #self._image_index = range(len(self._image_paths))
        self.classes = ['bg', 'face']
        self.num_classes = len(self.classes)


    def gt_roidb(self, min_area = 0):
        cache_file = os.path.join(self.cache_path, '{}_{}_gt_roidb.pkl'.format(self.name, self._split))
        # if not config.PROGRESSIVE:
        if not config.progressive_manual:
            if os.path.exists(cache_file):
                with open(cache_file, 'rb') as fid:
                    roidb = pickle.load(fid)
                print('{} gt roidb loaded from {}'.format(self.name, cache_file))
                self.num_images = len(roidb)
                return roidb

        roidb = []
        max_num_boxes = 0
        nonattr_box_num = 0
        landmark_num = 0

        for fp in self._fp_bbox_map:
            if self._split=='test':
              image_path = os.path.join(self._imgs_path, fp)
              roi = {'image': image_path}
              roidb.append(roi)
              continue
            boxes = np.zeros([len(self._fp_bbox_map[fp]), 4], np.float)
            landmarks = np.zeros([len(self._fp_bbox_map[fp]), 5, 3], np.float)
            blur = np.zeros((len(self._fp_bbox_map[fp]),), np.float)
            boxes_mask = []

            gt_classes = np.ones([len(self._fp_bbox_map[fp])], np.int32)
            overlaps = np.zeros([len(self._fp_bbox_map[fp]), 2], np.float)

            ix = 0

            for aline in self._fp_bbox_map[fp]:
                imsize = Image.open(os.path.join(self._imgs_path, fp)).size
                values = [float(x) for x in aline.strip().split()]
                bbox = [values[0], values[1], values[0]+values[2], values[1]+values[3]]

                x1 = bbox[0]
                y1 = bbox[1]
                x2 = min(imsize[0], bbox[2])
                y2 = min(imsize[1], bbox[3])
                if x1>=x2 or y1>=y2:
                  continue

                if config.BBOX_MASK_THRESH>0:
                  if (x2 - x1) < config.BBOX_MASK_THRESH or y2 - y1 < config.BBOX_MASK_THRESH:
                    boxes_mask.append(np.array([x1, y1, x2, y2], np.float))
                    continue
                if (x2 - x1) < config.TRAIN.MIN_BOX_SIZE or y2 - y1 < config.TRAIN.MIN_BOX_SIZE:
                    continue
                #if config.PROGRESSIVE:
                if config.progressive_manual:
                    area = (x2-x1)*(y2-y1)
                    if area < min_area:
                        continue

                boxes[ix, :] = np.array([x1, y1, x2, y2], np.float)
                if self._split=='train':
                  landmark = np.array( values[4:19], dtype=np.float32 ).reshape((5,3))
                  for li in range(5):
                    #print(landmark)
                    if landmark[li][0]==-1. and landmark[li][1]==-1.: #missing landmark
                      assert landmark[li][2]==-1
                    else:
                      assert landmark[li][2]>=0
                      if li==0:
                        landmark_num+=1
                      if landmark[li][2]==0.0:#visible
                        landmark[li][2] = 1.0
                      else:
                        landmark[li][2] = 0.0

                  landmarks[ix] = landmark

                  blur[ix] = values[19]
                  #print(aline, blur[ix])
                  if blur[ix]<0.0:
                    blur[ix] = 0.3
                    nonattr_box_num+=1

                cls = int(1)
                gt_classes[ix] = cls
                overlaps[ix, cls] = 1.0
                ix += 1
            max_num_boxes = max(max_num_boxes, ix)
            #overlaps = scipy.sparse.csr_matrix(overlaps)
            if self._split=='train' and ix==0:
              continue
            boxes = boxes[:ix,:]
            landmarks = landmarks[:ix,:,:]
            blur = blur[:ix]
            gt_classes = gt_classes[:ix]
            overlaps = overlaps[:ix,:]
            image_path = os.path.join(self._imgs_path, fp)
            with open(image_path, 'rb') as fin:
                stream = fin.read()
            stream = np.fromstring(stream, dtype=np.uint8)

            roi = {
              'image': image_path,
              'stream': stream,
              'height': imsize[1],
              'width': imsize[0],
              'boxes': boxes,
              'landmarks': landmarks,
              'blur': blur,
              'gt_classes': gt_classes,
              'gt_overlaps': overlaps,
              'max_classes': overlaps.argmax(axis=1),
              'max_overlaps': overlaps.max(axis=1),
              'flipped': False,
            }
            if len(boxes_mask)>0:
              boxes_mask = np.array(boxes_mask)
              roi['boxes_mask'] = boxes_mask
            roidb.append(roi)
        for roi in roidb:
          roi['max_num_boxes'] = max_num_boxes
        self.num_images = len(roidb)
        print('roidb size', len(roidb))
        print('non attr box num', nonattr_box_num)
        print('landmark num', landmark_num)
        # if not config.PROGRESSIVE:
        if not config.progressive_manual:
            with open(cache_file, 'wb') as fid:
                pickle.dump(roidb, fid, pickle.HIGHEST_PROTOCOL)
            print('wrote gt roidb to {}'.format(cache_file))

        return roidb

    def write_detections(self, all_boxes, output_dir='./output/'):
      pass

        
    def evaluate_detections(self, all_boxes, output_dir='./output/',method_name='insightdetection'):
      pass


