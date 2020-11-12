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
"""
This file has functions about generating bounding box regression targets
"""

from ..pycocotools.mask import encode
import numpy as np

from ..logger import logger
from .bbox_transform import bbox_overlaps, bbox_transform
from code.test.precisionTest.rcnn.config import config
import math
import cv2
import PIL.Image as Image
import threading
import Queue


def compute_bbox_regression_targets(rois, overlaps, labels):
    """
    given rois, overlaps, gt labels, compute bounding box regression targets
    :param rois: roidb[i]['boxes'] k * 4
    :param overlaps: roidb[i]['max_overlaps'] k * 1
    :param labels: roidb[i]['max_classes'] k * 1
    :return: targets[i][class, dx, dy, dw, dh] k * 5
    """
    # Ensure ROIs are floats
    rois = rois.astype(np.float, copy=False)

    # Sanity check
    if len(rois) != len(overlaps):
        logger.warning('bbox regression: len(rois) != len(overlaps)')

    # Indices of ground-truth ROIs
    gt_inds = np.where(overlaps == 1)[0]
    if len(gt_inds) == 0:
        logger.warning('bbox regression: len(gt_inds) == 0')

    # Indices of examples for which we try to make predictions
    ex_inds = np.where(overlaps >= config.TRAIN.BBOX_REGRESSION_THRESH)[0]

    # Get IoU overlap between each ex ROI and gt ROI
    ex_gt_overlaps = bbox_overlaps(rois[ex_inds, :], rois[gt_inds, :])

    # Find which gt ROI each ex ROI has max overlap with:
    # this will be the ex ROI's gt target
    gt_assignment = ex_gt_overlaps.argmax(axis=1)
    gt_rois = rois[gt_inds[gt_assignment], :]
    ex_rois = rois[ex_inds, :]

    targets = np.zeros((rois.shape[0], 5), dtype=np.float32)
    targets[ex_inds, 0] = labels[ex_inds]
    targets[ex_inds, 1:] = bbox_transform(ex_rois, gt_rois)
    return targets


def add_bbox_regression_targets(roidb):
    """
    given roidb, add ['bbox_targets'] and normalize bounding box regression targets
    :param roidb: roidb to be processed. must have gone through imdb.prepare_roidb
    :return: means, std variances of targets
    """
    logger.info('bbox regression: add bounding box regression targets')
    assert len(roidb) > 0
    assert 'max_classes' in roidb[0]

    num_images = len(roidb)
    num_classes = roidb[0]['gt_overlaps'].shape[1]
    for im_i in range(num_images):
        rois = roidb[im_i]['boxes']
        max_overlaps = roidb[im_i]['max_overlaps']
        max_classes = roidb[im_i]['max_classes']
        roidb[im_i]['bbox_targets'] = compute_bbox_regression_targets(rois, max_overlaps, max_classes)

    if config.TRAIN.BBOX_NORMALIZATION_PRECOMPUTED:
        # use fixed / precomputed means and stds instead of empirical values
        means = np.tile(np.array(config.TRAIN.BBOX_MEANS), (num_classes, 1))
        stds = np.tile(np.array(config.TRAIN.BBOX_STDS), (num_classes, 1))
    else:
        # compute mean, std values
        class_counts = np.zeros((num_classes, 1)) + 1e-14
        sums = np.zeros((num_classes, 4))
        squared_sums = np.zeros((num_classes, 4))
        for im_i in range(num_images):
            targets = roidb[im_i]['bbox_targets']
            for cls in range(1, num_classes):
                cls_indexes = np.where(targets[:, 0] == cls)[0]
                if cls_indexes.size > 0:
                    class_counts[cls] += cls_indexes.size
                    sums[cls, :] += targets[cls_indexes, 1:].sum(axis=0)
                    squared_sums[cls, :] += (targets[cls_indexes, 1:] ** 2).sum(axis=0)

        means = sums / class_counts
        # var(x) = E(x^2) - E(x)^2
        stds = np.sqrt(squared_sums / class_counts - means ** 2)

    # normalized targets
    for im_i in range(num_images):
        targets = roidb[im_i]['bbox_targets']
        for cls in range(1, num_classes):
            cls_indexes = np.where(targets[:, 0] == cls)[0]
            roidb[im_i]['bbox_targets'][cls_indexes, 1:] -= means[cls, :]
            roidb[im_i]['bbox_targets'][cls_indexes, 1:] /= stds[cls, :]

    return means.ravel(), stds.ravel()


def expand_bbox_regression_targets(bbox_targets_data, num_classes):
    """
    expand from 5 to 4 * num_classes; only the right class has non-zero bbox regression targets
    :param bbox_targets_data: [k * 5]
    :param num_classes: number of classes
    :return: bbox target processed [k * 4 num_classes]
    bbox_weights ! only foreground boxes have bbox regression computation!
    """
    classes = bbox_targets_data[:, 0]
    bbox_targets = np.zeros((classes.size, 4 * num_classes), dtype=np.float32)
    bbox_weights = np.zeros(bbox_targets.shape, dtype=np.float32)
    indexes = np.where(classes > 0)[0]
    for index in indexes:
        cls = classes[index]
        start = int(4 * cls)
        end = start + 4
        bbox_targets[index, start:end] = bbox_targets_data[index, 1:]
        bbox_weights[index, start:end] = config.TRAIN.BBOX_WEIGHTS
    return bbox_targets, bbox_weights


def compute_mask_and_label(ex_rois, ex_labels, seg, flipped):
    # assert os.path.exists(seg_gt), 'Path does not exist: {}'.format(seg_gt)
    # im = Image.open(seg_gt)
    # pixel = list(im.getdata())
    # pixel = np.array(pixel).reshape([im.size[1], im.size[0]])
    im = Image.open(seg)
    pixel = list(im.getdata())
    ins_seg = np.array(pixel).reshape([im.size[1], im.size[0]])
    if flipped:
        ins_seg = ins_seg[:, ::-1]
    rois = ex_rois
    n_rois = ex_rois.shape[0]
    label = ex_labels
    class_id = config.CLASS_ID
    mask_target = np.zeros((n_rois, 28, 28), dtype=np.int8)
    mask_label = np.zeros((n_rois), dtype=np.int8)
    for n in range(n_rois):
        target = ins_seg[int(rois[n, 1]): int(rois[n, 3]), int(rois[n, 0]): int(rois[n, 2])]
        ids = np.unique(target)
        ins_id = 0
        max_count = 0
        for id in ids:
            if math.floor(id / 1000) == class_id[int(label[int(n)])]:
                px = np.where(ins_seg == int(id))
                x_min = np.min(px[1])
                y_min = np.min(px[0])
                x_max = np.max(px[1])
                y_max = np.max(px[0])
                x1 = max(rois[n, 0], x_min)
                y1 = max(rois[n, 1], y_min)
                x2 = min(rois[n, 2], x_max)
                y2 = min(rois[n, 3], y_max)
                iou = (x2 - x1) * (y2 - y1)
                iou = iou / ((rois[n, 2] - rois[n, 0]) * (rois[n, 3] - rois[n, 1])
                             + (x_max - x_min) * (y_max - y_min) - iou)
                if iou > max_count:
                    ins_id = id
                    max_count = iou

        if max_count == 0:
            continue
        # print max_count
        mask = np.zeros(target.shape)
        idx = np.where(target == ins_id)
        mask[idx] = 1
        mask = cv2.resize(mask, (28, 28), interpolation=cv2.INTER_NEAREST)

        mask_target[n] = mask
        mask_label[n] = label[int(n)]
    return mask_target, mask_label


def compute_bbox_mask_targets_and_label(rois, overlaps, labels, seg, flipped):
    """
    given rois, overlaps, gt labels, seg, compute bounding box mask targets
    :param rois: roidb[i]['boxes'] k * 4
    :param overlaps: roidb[i]['max_overlaps'] k * 1
    :param labels: roidb[i]['max_classes'] k * 1
    :return: targets[i][class, dx, dy, dw, dh] k * 5
    """
    # Ensure ROIs are floats
    rois = rois.astype(np.float, copy=False)

    # Sanity check
    if len(rois) != len(overlaps):
        print('bbox regression: this should not happen')

    # Indices of ground-truth ROIs
    gt_inds = np.where(overlaps == 1)[0]
    if len(gt_inds) == 0:
        print('something wrong : zero ground truth rois')
    # Indices of examples for which we try to make predictions
    ex_inds = np.where(overlaps >= config.TRAIN.BBOX_REGRESSION_THRESH)[0]

    # Get IoU overlap between each ex ROI and gt ROI
    ex_gt_overlaps = bbox_overlaps(rois[ex_inds, :], rois[gt_inds, :])


    # Find which gt ROI each ex ROI has max overlap with:
    # this will be the ex ROI's gt target
    gt_assignment = ex_gt_overlaps.argmax(axis=1)
    gt_rois = rois[gt_inds[gt_assignment], :]
    ex_rois = rois[ex_inds, :]

    mask_targets, mask_label = compute_mask_and_label(ex_rois, labels[ex_inds], seg, flipped)
    return mask_targets, mask_label, ex_inds

def add_mask_targets(roidb):
    """
    given roidb, add ['bbox_targets'] and normalize bounding box regression targets
    :param roidb: roidb to be processed. must have gone through imdb.prepare_roidb
    :return: means, std variances of targets
    """
    print('add bounding box mask targets')
    assert len(roidb) > 0
    assert 'max_classes' in roidb[0]

    num_images = len(roidb)

    # Multi threads processing
    im_quene = Queue.Queue(maxsize=0)
    for im_i in range(num_images):
        im_quene.put(im_i)

    def process():
        while not im_quene.empty():
            im_i = im_quene.get()
            print("-----process img {}".format(im_i))
            rois = roidb[im_i]['boxes']
            max_overlaps = roidb[im_i]['max_overlaps']
            max_classes = roidb[im_i]['max_classes']
            ins_seg = roidb[im_i]['ins_seg']
            flipped = roidb[im_i]['flipped']
            roidb[im_i]['mask_targets'], roidb[im_i]['mask_labels'], roidb[im_i]['mask_inds'] = \
                compute_bbox_mask_targets_and_label(rois, max_overlaps, max_classes, ins_seg, flipped)
    threads = [threading.Thread(target=process, args=()) for i in xrange(10)]
    for t in threads: t.start()
    for t in threads: t.join()
    # Single thread
    # for im_i in range(num_images):
    #     print "-----processing img {}".format(im_i)
    #     rois = roidb[im_i]['boxes']
    #     max_overlaps = roidb[im_i]['max_overlaps']
    #     max_classes = roidb[im_i]['max_classes']
    #     ins_seg = roidb[im_i]['ins_seg']
    #     # roidb[im_i]['mask_targets'] = compute_bbox_mask_targets(rois, max_overlaps, max_classes, ins_seg)
    #     roidb[im_i]['mask_targets'], roidb[im_i]['mask_labels'], roidb[im_i]['mask_inds'] = \
    #         compute_bbox_mask_targets_and_label(rois, max_overlaps, max_classes, ins_seg)
