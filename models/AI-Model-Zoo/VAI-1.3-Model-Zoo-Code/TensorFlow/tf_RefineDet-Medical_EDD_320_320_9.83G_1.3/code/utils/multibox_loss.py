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


#coding=utf-8

from utils.box_utils import * 

def modified_smooth_l1(bbox_pred, bbox_targets, bbox_inside_weights=1., bbox_outside_weights=1., sigma=1.):
    sigma2 = sigma * sigma
    inside_mul = tf.multiply(bbox_inside_weights, tf.subtract(bbox_pred, bbox_targets))
    smooth_l1_sign = tf.cast(tf.less(tf.abs(inside_mul), 1.0 / sigma2), tf.float32)
    smooth_l1_option1 = tf.multiply(tf.multiply(inside_mul, inside_mul), 0.5 * sigma2)
    smooth_l1_option2 = tf.subtract(tf.abs(inside_mul), 0.5 / sigma2)
    smooth_l1_result = tf.add(tf.multiply(smooth_l1_option1, smooth_l1_sign),
                              tf.multiply(smooth_l1_option2, tf.abs(tf.subtract(smooth_l1_sign, 1.0))))
    outside_mul = tf.multiply(bbox_outside_weights, smooth_l1_result)
    return outside_mul

def multibox_loss(predictions, gt_labels, gt_bboxes, num_reals, num_class, priorboxes, variances, iou_threshold=0.5, ohem=True, negative_ratio=3, use_arm=False, filter_obj_score=0.01, minnum_negative=3):
    if use_arm:
        arm_cls, arm_loc, odm_cls, odm_loc = predictions
        conf_prd, loc_prd = odm_cls, odm_loc

        def refine_match_fn(arm_loc, gt_labels, gt_bboxes, num_real, priorboxes=priorboxes, variances=variances, iou_threshold=iou_threshold):
            gt_labels = tf.slice(gt_labels, [0], [num_real])
            gt_bboxes = tf.slice(gt_bboxes, [0, 0], [num_real, 4])

            priorboxes_refine = to_center(decode(arm_loc, priorboxes, variances))
            conf_t, loc_t = match(gt_labels, gt_bboxes, priorboxes_refine, variances, iou_threshold)
            return conf_t, loc_t
        
        lamb_refine_match_fn = lambda x: refine_match_fn(x[0], x[1], x[2], x[3])               
        gt_targets = tf.map_fn(lamb_refine_match_fn, (arm_loc, gt_labels, gt_bboxes, num_reals), dtype=(tf.int64, tf.float32), back_prop=False)
           
    else:
        arm_cls, arm_loc = predictions
        conf_prd, loc_prd = arm_cls, arm_loc
        
        def match_fn(gt_labels, gt_bboxes, num_real, priorboxes=priorboxes, variances=variances, iou_threshold=iou_threshold, use_binary_class=True):
            gt_labels = tf.slice(gt_labels, [0], [num_real])
            gt_bboxes = tf.slice(gt_bboxes, [0, 0], [num_real, 4])
            if use_binary_class:
                gt_labels = tf.ones_like(gt_labels, dtype=tf.int64) 
            conf_t, loc_t = match(gt_labels, gt_bboxes, priorboxes, variances, iou_threshold)
            return conf_t, loc_t

        num_class = 2
        lamb_match_fn = lambda x: match_fn(x[0], x[1], x[2])
        gt_targets = tf.map_fn(lamb_match_fn, (gt_labels, gt_bboxes, num_reals), dtype=(tf.int64, tf.float32), back_prop=False)
         
    conf_targets = tf.concat(gt_targets[0], axis=0)
    loc_targets = tf.concat(gt_targets[1], axis=0)

    flaten_conf_targets = tf.reshape(conf_targets, [-1])
    flaten_loc_targets = tf.reshape(loc_targets, [-1, 4])
    flaten_conf_targets = tf.stop_gradient(flaten_conf_targets)
    flaten_loc_targets = tf.stop_gradient(flaten_loc_targets)
   
    flaten_conf_pred = tf.reshape(conf_prd, [-1, num_class])
    flaten_loc_pred = tf.reshape(loc_prd, [-1, 4])

    positive_mask = tf.stop_gradient(flaten_conf_targets > 0)

    if ohem:
        batch_n_positives = tf.count_nonzero(conf_targets, -1)
        batch_negtive_mask = tf.equal(conf_targets, 0) 

        if use_arm:
            prob_for_obj = tf.nn.softmax(arm_cls)[:, :, 1]
            arm_filter_mask = tf.greater_equal(prob_for_obj, filter_obj_score)
            batch_negtive_mask = tf.logical_and(batch_negtive_mask, arm_filter_mask)

        batch_n_negtives = tf.count_nonzero(batch_negtive_mask, -1)
        batch_n_neg_select = tf.minimum(tf.cast(negative_ratio * tf.cast(batch_n_positives, tf.float32), tf.int32), tf.cast(batch_n_negtives, tf.int32))
        batch_n_neg_select = tf.where(batch_n_neg_select >= minnum_negative, batch_n_neg_select, tf.ones_like(batch_n_neg_select) * minnum_negative)

        prob_for_bg = tf.nn.softmax(tf.reshape(conf_prd, [tf.shape(gt_labels)[0], -1, num_class]))[:, :, 0]
        prob_for_negtives = tf.where(batch_negtive_mask,
                                     0. - prob_for_bg,
                                     0. - tf.ones_like(prob_for_bg))
        topk_prob_for_bg, _ = tf.nn.top_k(prob_for_negtives, k=tf.shape(prob_for_negtives)[1])
        score_at_k = tf.gather_nd(topk_prob_for_bg, tf.stack([tf.range(tf.shape(gt_labels)[0]), batch_n_neg_select - 1], axis=-1))
        selected_neg_mask = prob_for_negtives >= tf.expand_dims(score_at_k, axis=-1)

        final_select_mask = tf.stop_gradient(tf.logical_or(tf.reshape(tf.logical_and(batch_negtive_mask, selected_neg_mask), [-1]), positive_mask))
        #total_examples = tf.count_nonzero(final_select_mask)
        flaten_conf_pred_select = tf.boolean_mask(flaten_conf_pred, final_select_mask)
        flaten_loc_pred_select = tf.boolean_mask(flaten_loc_pred, positive_mask)
        flaten_conf_targets_select = tf.boolean_mask(flaten_conf_targets, final_select_mask)
        flaten_loc_targets_select = tf.boolean_mask(flaten_loc_targets, positive_mask)

        conf_loss = tf.losses.sparse_softmax_cross_entropy(labels=flaten_conf_targets_select, logits=flaten_conf_pred_select)
        loc_loss = modified_smooth_l1(flaten_loc_pred_select, flaten_loc_targets_select, sigma=1.)
         
    else:
        flaten_loc_pred_select = tf.boolean_mask(flaten_loc_pred, positive_mask)
        flaten_loc_targets_select = tf.boolean_mask(flaten_loc_targets, positive_mask)

        conf_loss = tf.losses.sparse_softmax_cross_entropy(labels=flaten_conf_targets, logits=flaten_conf_pred) 
        loc_loss = modified_smooth_l1(flaten_loc_pred_select, flaten_loc_targets_select, sigma=1.)

    loc_loss = tf.reduce_mean(tf.reduce_sum(loc_loss, axis=-1), name='loc_loss')
    return conf_loss, loc_loss
