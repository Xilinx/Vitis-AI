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

import numpy as np
from ..cython.bbox import bbox_overlaps_cython
from ..config import config
#from rcnn.config import config


def bbox_overlaps(boxes, query_boxes):
    return bbox_overlaps_cython(boxes, query_boxes)


def bbox_overlaps_py(boxes, query_boxes):
    """
    determine overlaps between boxes and query_boxes
    :param boxes: n * 4 bounding boxes
    :param query_boxes: k * 4 bounding boxes
    :return: overlaps: n * k overlaps
    """
    n_ = boxes.shape[0]
    k_ = query_boxes.shape[0]
    overlaps = np.zeros((n_, k_), dtype=np.float)
    for k in range(k_):
        query_box_area = (query_boxes[k, 2] - query_boxes[k, 0] + 1) * (query_boxes[k, 3] - query_boxes[k, 1] + 1)
        for n in range(n_):
            iw = min(boxes[n, 2], query_boxes[k, 2]) - max(boxes[n, 0], query_boxes[k, 0]) + 1
            if iw > 0:
                ih = min(boxes[n, 3], query_boxes[k, 3]) - max(boxes[n, 1], query_boxes[k, 1]) + 1
                if ih > 0:
                    box_area = (boxes[n, 2] - boxes[n, 0] + 1) * (boxes[n, 3] - boxes[n, 1] + 1)
                    all_area = float(box_area + query_box_area - iw * ih)
                    overlaps[n, k] = iw * ih / all_area
    return overlaps


def clip_boxes(boxes, im_shape):
    """
    Clip boxes to image boundaries.
    :param boxes: [N, 4* num_classes]
    :param im_shape: tuple of 2
    :return: [N, 4* num_classes]
    """
    # x1 >= 0
    boxes[:, 0::4] = np.maximum(np.minimum(boxes[:, 0::4], im_shape[1] - 1), 0)
    # y1 >= 0
    boxes[:, 1::4] = np.maximum(np.minimum(boxes[:, 1::4], im_shape[0] - 1), 0)
    # x2 < im_shape[1]
    boxes[:, 2::4] = np.maximum(np.minimum(boxes[:, 2::4], im_shape[1] - 1), 0)
    # y2 < im_shape[0]
    boxes[:, 3::4] = np.maximum(np.minimum(boxes[:, 3::4], im_shape[0] - 1), 0)
    return boxes


def nonlinear_transform(ex_rois, gt_rois):
    """
    compute bounding box regression targets from ex_rois to gt_rois
    :param ex_rois: [N, 4]
    :param gt_rois: [N, 4]
    :return: [N, 4]
    """
    assert ex_rois.shape[0] == gt_rois.shape[0], 'inconsistent rois number'

    ex_widths = ex_rois[:, 2] - ex_rois[:, 0] + 1.0
    ex_heights = ex_rois[:, 3] - ex_rois[:, 1] + 1.0
    ex_ctr_x = ex_rois[:, 0] + 0.5 * (ex_widths - 1.0)
    ex_ctr_y = ex_rois[:, 1] + 0.5 * (ex_heights - 1.0)

    gt_widths = gt_rois[:, 2] - gt_rois[:, 0] + 1.0
    gt_heights = gt_rois[:, 3] - gt_rois[:, 1] + 1.0
    gt_ctr_x = gt_rois[:, 0] + 0.5 * (gt_widths - 1.0)
    gt_ctr_y = gt_rois[:, 1] + 0.5 * (gt_heights - 1.0)

    targets_dx = (gt_ctr_x - ex_ctr_x) / (ex_widths + 1e-14)
    targets_dy = (gt_ctr_y - ex_ctr_y) / (ex_heights + 1e-14)
    targets_dw = np.log(gt_widths / ex_widths)
    targets_dh = np.log(gt_heights / ex_heights)

    if gt_rois.shape[1]<=4:
      targets = np.vstack(
          (targets_dx, targets_dy, targets_dw, targets_dh)).transpose()
      return targets
    else:
      targets = [targets_dx, targets_dy, targets_dw, targets_dh]
      #if config.USE_BLUR:
      #  for i in range(4, gt_rois.shape[1]):
      #    t = gt_rois[:,i]
      #    targets.append(t)
      targets = np.vstack(targets).transpose()
      return targets

def landmark_transform(ex_rois, gt_rois):

    assert ex_rois.shape[0] == gt_rois.shape[0], 'inconsistent rois number'

    ex_widths = ex_rois[:, 2] - ex_rois[:, 0] + 1.0
    ex_heights = ex_rois[:, 3] - ex_rois[:, 1] + 1.0
    ex_ctr_x = ex_rois[:, 0] + 0.5 * (ex_widths - 1.0)
    ex_ctr_y = ex_rois[:, 1] + 0.5 * (ex_heights - 1.0)

    
    targets = []
    for i in range(gt_rois.shape[1]):
      for j in range(gt_rois.shape[2]):
        #if not config.USE_OCCLUSION and j==2:
        #  continue
        if j==2:
          continue
        if j==0: #w
          target = (gt_rois[:,i,j] - ex_ctr_x) / (ex_widths + 1e-14)
        elif j==1: #h
          target = (gt_rois[:,i,j] - ex_ctr_y) / (ex_heights + 1e-14)
        else: #visibile
          target = gt_rois[:,i,j]
        targets.append(target)


    targets = np.vstack(targets).transpose()
    return targets


def landmark_transform_af(ct_int,gt_roi,gt_box,output_h,output_w):

    gt_widths = gt_box[0, 2] - gt_box[0, 0]
    gt_heights = gt_box[0, 3] - gt_box[0, 1]

    targets = np.zeros((5,2),dtype=np.float32)
    for i in range(gt_roi.shape[1]):
        for j in range(gt_roi.shape[2]):
            # if not config.USE_OCCLUSION and j==2:
            #  continue
            if j == 2:
                continue
            if j == 0:  # w
                #target = (gt_roi[:, i, j] - ct_int[0]) / (gt_widths + 1e-14)
                targets[i,j] = (gt_roi[0, i, j] - ct_int[0]) / (gt_widths + 1e-14)
            elif j == 1:  # h
                #target = (gt_roi[:, i, j] - ct_int[1]) / (gt_heights + 1e-14)
                targets[i,j] = (gt_roi[0, i, j] - ct_int[1]) / (gt_heights + 1e-14)

    #targets = np.vstack(targets).transpose()
    return targets
def nonlinear_pred(boxes, box_deltas):
    """
    Transform the set of class-agnostic boxes into class-specific boxes
    by applying the predicted offsets (box_deltas)
    :param boxes: !important [N 4]
    :param box_deltas: [N, 4 * num_classes]
    :return: [N 4 * num_classes]
    """
    if boxes.shape[0] == 0:
        return np.zeros((0, box_deltas.shape[1]))

    boxes = boxes.astype(np.float, copy=False)
    widths = boxes[:, 2] - boxes[:, 0] + 1.0
    heights = boxes[:, 3] - boxes[:, 1] + 1.0
    ctr_x = boxes[:, 0] + 0.5 * (widths - 1.0)
    ctr_y = boxes[:, 1] + 0.5 * (heights - 1.0)

    dx = box_deltas[:, 0::4]
    dy = box_deltas[:, 1::4]
    dw = box_deltas[:, 2::4]
    dh = box_deltas[:, 3::4]

    pred_ctr_x = dx * widths[:, np.newaxis] + ctr_x[:, np.newaxis]
    pred_ctr_y = dy * heights[:, np.newaxis] + ctr_y[:, np.newaxis]
    pred_w = np.exp(dw) * widths[:, np.newaxis]
    pred_h = np.exp(dh) * heights[:, np.newaxis]

    pred_boxes = np.zeros(box_deltas.shape)
    # x1
    pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * (pred_w - 1.0)
    # y1
    pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * (pred_h - 1.0)
    # x2
    pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * (pred_w - 1.0)
    # y2
    pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * (pred_h - 1.0)

    return pred_boxes

def landmark_pred(boxes, landmark_deltas):
    if boxes.shape[0] == 0:
        return np.zeros((0, landmark_deltas.shape[1]))
    boxes = boxes.astype(np.float, copy=False)
    widths = boxes[:, 2] - boxes[:, 0] + 1.0
    heights = boxes[:, 3] - boxes[:, 1] + 1.0
    ctr_x = boxes[:, 0] + 0.5 * (widths - 1.0)
    ctr_y = boxes[:, 1] + 0.5 * (heights - 1.0)
    preds = []
    for i in range(landmark_deltas.shape[1]):
      if i%2==0:
        pred = (landmark_deltas[:,i]*widths + ctr_x)
      else:
        pred = (landmark_deltas[:,i]*heights + ctr_y)
      preds.append(pred)
    preds = np.vstack(preds).transpose()
    return preds

def iou_transform(ex_rois, gt_rois):
    """ return bbox targets, IoU loss uses gt_rois as gt """
    assert ex_rois.shape[0] == gt_rois.shape[0], 'inconsistent rois number'
    return gt_rois


def iou_pred(boxes, box_deltas):
    """
    Transform the set of class-agnostic boxes into class-specific boxes
    by applying the predicted offsets (box_deltas)
    :param boxes: !important [N 4]
    :param box_deltas: [N, 4 * num_classes]
    :return: [N 4 * num_classes]
    """
    if boxes.shape[0] == 0:
        return np.zeros((0, box_deltas.shape[1]))

    boxes = boxes.astype(np.float, copy=False)
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    dx1 = box_deltas[:, 0::4]
    dy1 = box_deltas[:, 1::4]
    dx2 = box_deltas[:, 2::4]
    dy2 = box_deltas[:, 3::4]

    pred_boxes = np.zeros(box_deltas.shape)
    # x1
    pred_boxes[:, 0::4] = dx1 + x1[:, np.newaxis]
    # y1
    pred_boxes[:, 1::4] = dy1 + y1[:, np.newaxis]
    # x2
    pred_boxes[:, 2::4] = dx2 + x2[:, np.newaxis]
    # y2
    pred_boxes[:, 3::4] = dy2 + y2[:, np.newaxis]

    return pred_boxes

def bbox_transform_xyxy(ex_rois, gt_rois, weights=(1.0, 1.0, 1.0, 1.0)):
    #  0 1 2 3
    #  < v > ^

    gt_l = gt_rois[:, 0]
    gt_r = gt_rois[:, 2]
    gt_d = gt_rois[:, 1]
    gt_u = gt_rois[:, 3]

    ex_l = ex_rois[:, 0]
    ex_r = ex_rois[:, 2]
    ex_d = ex_rois[:, 1]
    ex_u = ex_rois[:, 3]

    ex_widths = ex_r - ex_l + 1.0
    ex_heights = ex_u - ex_d + 1.0

    wx, wy, ww, wh = weights
    targets_dl = wx * (gt_l - ex_l) / ex_widths
    targets_dr = wy * (gt_r - ex_r) / ex_widths
    targets_dd = wx * (gt_d - ex_d) / ex_heights
    targets_du = wy * (gt_u - ex_u) / ex_heights

    targets = np.vstack(
        (targets_dl, targets_dr, targets_dd, targets_du)).transpose()
    return targets


def landmark_transform_xyxy(ex_rois, gt_rois):
    assert ex_rois.shape[0] == gt_rois.shape[0], 'inconsistent rois number'

    ex_l = ex_rois[:, 0]
    ex_r = ex_rois[:, 2]
    ex_d = ex_rois[:, 1]
    ex_u = ex_rois[:, 3]

    ex_widths = ex_r - ex_l + 1.0
    ex_heights = ex_u - ex_d + 1.0

    targets = []
    for i in range(gt_rois.shape[1]):
        for j in range(gt_rois.shape[2]):
            # if not config.USE_OCCLUSION and j==2:
            #  continue
            if j == 2:
                continue
            if j == 0:  # w
                if i % 3 == 0 or i % 3 == 2:
                    target = (gt_rois[:, i, j] - ex_l) / (ex_widths + 1e-14)
                elif i % 3 == 1:
                    target = (gt_rois[:, i, j] - ex_r) / (ex_widths + 1e-14)

            elif j == 1:  # h
                if i % 3 == 0 or i % 3 == 2:
                    target = (gt_rois[:, i, j] - ex_d) / (ex_heights + 1e-14)
                elif i % 3 == 1:
                    target = (gt_rois[:, i, j] - ex_u) / (ex_heights + 1e-14)
            else:  # visibile
                target = gt_rois[:, i, j]
            targets.append(target)

    targets = np.vstack(targets).transpose()
    return targets

def bbox_pred_xyxy(boxes, deltas, weights=(1.0, 1.0, 1.0, 1.0)):
    if boxes.shape[0] == 0:
        return np.zeros((0, deltas.shape[1]), dtype=deltas.dtype)

    boxes = boxes.astype(deltas.dtype, copy=False)

    #  0 1 2 3
    #  < v > ^

    widths = boxes[:, 2] - boxes[:, 0] + 1.0
    heights = boxes[:, 3] - boxes[:, 1] + 1.0

    l = boxes[:, 0]
    r = boxes[:, 2]
    d = boxes[:, 1]
    u = boxes[:, 3]

    wx, wy, ww, wh = weights
    dl = deltas[:, 0::4] / wx
    dr = deltas[:, 1::4] / wy
    dd = deltas[:, 2::4] / wx
    du = deltas[:, 3::4] / wy
    # Prevent sending too large values into np.exp()
    BBOX_XFORM_CLIP = np.log(1000. / 16.)
    BBOX_XFORM_CLIPe = 1000. / 16.
    dl = np.maximum(np.minimum(dl, BBOX_XFORM_CLIPe), -BBOX_XFORM_CLIPe)
    dr = np.maximum(np.minimum(dr, BBOX_XFORM_CLIPe), -BBOX_XFORM_CLIPe)
    dd = np.maximum(np.minimum(dd, BBOX_XFORM_CLIPe), -BBOX_XFORM_CLIPe)
    du = np.maximum(np.minimum(du, BBOX_XFORM_CLIPe), -BBOX_XFORM_CLIPe)

    pred_l = dl * widths[:, np.newaxis] + l[:, np.newaxis]
    pred_r = dr * widths[:, np.newaxis] + r[:, np.newaxis]
    pred_d = dd * heights[:, np.newaxis] + d[:, np.newaxis]
    pred_u = du * heights[:, np.newaxis] + u[:, np.newaxis]

    pred_boxes = np.zeros(deltas.shape, dtype=deltas.dtype)
    # x1
    pred_boxes[:, 0::4] = pred_l
    # y1
    pred_boxes[:, 1::4] = pred_d
    # x2
    pred_boxes[:, 2::4] = pred_r
    # y2
    pred_boxes[:, 3::4] = pred_u

    return pred_boxes

# define bbox_transform and bbox_pred
bbox_transform = nonlinear_transform
if config.USE_KLLOSS:
    bbox_transform = bbox_transform_xyxy
bbox_pred = nonlinear_pred
if config.USE_KLLOSS:
    bbox_pred = bbox_pred_xyxy

