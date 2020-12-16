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



from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

#from blocks_utils import *
from model.vgg_bn import *
from utils.en_decode  import * 
from utils.prior_box import *
from utils.config import priorbox_config, detect_config 
from utils.multibox_loss import *
from utils.box_utils import *

class Refinedet(object):
   
    def __init__(self, num_class, data_format='channels_last', priorbox_config=priorbox_config, detect_config=detect_config):
        super(Refinedet, self).__init__()

        self._num_class = num_class
        self._data_format = data_format

        self._priorbox_config = priorbox_config 
        self._priorboxes_obj = Priorbox(self._priorbox_config.input_shape, self._priorbox_config.feature_shapes,
                                        self._priorbox_config.min_sizes, self._priorbox_config.max_sizes,
                                        self._priorbox_config.aspect_ratios, self._priorbox_config.steps,
                                        self._priorbox_config.offset)
        self._priorboxes = self._priorboxes_obj()
        self._variances = self._priorbox_config.variances
        self._num_pirors_depth_per_layer = self._priorboxes_obj.num_priors_depth_per_layer
        self._detect_config = detect_config
         
        self._feature_extracter = VGG_BN_Backbone(data_format=self._data_format)           
        self._extra_layers = self.__extra_layers(data_format=self._data_format, name='extra_layers')
        self._tcb_layers = self.__tcb_layers(data_format=self._data_format, name='tcb')
        self._arm_heads = self.__multibox_heads(2, self._num_pirors_depth_per_layer, data_format=self._data_format, name='arm') 
        self._odm_heads = self.__multibox_heads(self._num_class, self._num_pirors_depth_per_layer, data_format=self._data_format, name='odm') 
       
    def __extra_layers(self, data_format='channels_last', name=None):
        extra_layers = []
        with tf.variable_scope(name) as scope:
            extra_layers.append(ssd_conv_bn_block(256, 2, data_format=data_format, name='conv8'))
            extra_layers.append(ssd_conv_bn_block(128, 2, data_format=data_format, name='conv9'))

        return extra_layers

    def __tcb_layers(self, data_format='channels_last', name=None):
        tcb_layers = []
        with tf.variable_scope(name) as scope:
            tcb_layers.append(transfer_connection_block(data_format=data_format, name='conv4_bn_block_tcb'))
            tcb_layers.append(transfer_connection_block(data_format=data_format, name='fc7_m_tcb')) 
            tcb_layers.append(transfer_connection_block(data_format=data_format, name='conv8_tcb'))
            tcb_layers.append(transfer_connection_block(data_format=data_format, has_deconv_layer=False, name='conv9_tcb'))

        return tcb_layers

    def __multibox_heads(self, num_class, num_pirors_depth_per_layer, padding='same', data_format='channels_last', use_bias=True, name=None):
        cls_pred = []
        loc_pred = []
        with tf.variable_scope(name) as scope:
            for idx, num_prior in enumerate(num_pirors_depth_per_layer):
                cls_pred.append(tf.layers.Conv2D(filters=num_prior * num_class, kernel_size=3, strides=1, padding=padding,
                                                 data_format=data_format, activation=None, use_bias=use_bias,
                                                 kernel_initializer=conv_initializer(),
                                                 bias_initializer=tf.zeros_initializer(),
                                                 name='cls_{}'.format(idx), _scope='cls_{}'.format(idx),
                                                 _reuse=None))
                loc_pred.append(tf.layers.Conv2D(filters=num_prior * 4, kernel_size=3, strides=1, padding=padding,
                                                 data_format=data_format, activation=None, use_bias=use_bias,
                                                 kernel_initializer=conv_initializer(),
                                                 bias_initializer=tf.zeros_initializer(),
                                                 name='loc_{}'.format(idx), _scope='loc_{}'.format(idx),
                                                 _reuse=None))
            return cls_pred, loc_pred

    def __odm_fpn(self, arm_features, tcb_layers, use_bn_tcb=False, training=False):
        assert len(arm_features) == len(tcb_layers), 'arm detect head number must equal to odm detect head number!'
        odm_features = []
        concat_idx = 2 if not use_bn_tcb else 4 

        for feature, tcb_block in list(zip(arm_features, tcb_layers))[::-1]:
            if 'upsample' not in tcb_block.keys():
                feature = forward_block(feature, tcb_block['tcb'], training=training)
            else:
                tcb_block_concat_before = tcb_block['tcb'][:concat_idx + 1]
                feature_concat_before = forward_block(feature, tcb_block_concat_before, training=training)  

                upsample_block = tcb_block['upsample']
                feature_up_concat_before = forward_block(odm_features[-1], upsample_block, training=training)

                feature_concat = feature_concat_before + feature_up_concat_before
                tcb_block_concat_after = tcb_block['tcb'][concat_idx + 1:]     
                feature = forward_block(feature_concat, tcb_block_concat_after, training=training) 
            odm_features.append(feature)

        return odm_features[::-1]
     
    def __pred_head(self, features, heads, training=False):
        pred_cls = []
        pred_loc = []
        head_cls, head_loc = heads

        forward_predict = lambda x: forward_block(x[0], x[1], training=training)

        for idx, feature in enumerate(features):
            pred_cls.append(forward_predict([feature, head_cls[idx]]))
            pred_loc.append(forward_predict([feature, head_loc[idx]]))
      
        return pred_cls, pred_loc

    def forward(self, inputs, training=False):
        arm_features = []

        extract_features = self._feature_extracter.forward(inputs, training=training)    
        _, inputs = extract_features        
        arm_features.extend(extract_features)
        
        for extra_layer_block in self._extra_layers:
            inputs = forward_block(inputs, extra_layer_block, training=training)
            arm_features.append(inputs)

        odm_features = self.__odm_fpn(arm_features, self._tcb_layers, training=training)
        
        arm_pred_cls, arm_pred_loc = self.__pred_head(arm_features, self._arm_heads, training=training) 
        odm_pred_cls, odm_pred_loc = self.__pred_head(odm_features, self._odm_heads, training=training) 

        return arm_pred_cls, arm_pred_loc, odm_pred_cls, odm_pred_loc

    def predictions_transform(self, features, num_class):
        if self._data_format == 'channels_first':
            features = [tf.transpose(feature, [0, 2, 3, 1]) for feature in features]   
        features = [tf.reshape(feature, [tf.shape(feature)[0], -1, num_class]) for feature in features]
        return tf.concat(features, axis=1)        

    def refinedet_multibox_loss(self, predictions, targets, ohem=True, negative_ratio=3):
        gt_labels, gt_bboxes, num_reals = targets
        arm_cls, arm_loc, odm_cls, odm_loc = predictions

        arm_cls = self.predictions_transform(arm_cls, 2)
        arm_loc = self.predictions_transform(arm_loc, 4)
        odm_cls = self.predictions_transform(odm_cls, self._num_class)
        odm_loc = self.predictions_transform(odm_loc, 4)
       
        with tf.name_scope('arm_loss'):
            arm_conf_loss, arm_loc_loss = multibox_loss([arm_cls, arm_loc], gt_labels, gt_bboxes, num_reals, self._num_class, self._priorboxes, self._variances) 
        with tf.name_scope('odm_loss'):
            odm_conf_loss, odm_loc_loss = multibox_loss([arm_cls, arm_loc, odm_cls, odm_loc], gt_labels, gt_bboxes, num_reals, self._num_class, self._priorboxes, self._variances, iou_threshold=0.65, use_arm=True) 
        
        return arm_conf_loss * (1 + negative_ratio), arm_loc_loss, odm_conf_loss * (1 + negative_ratio), odm_loc_loss

    def detect(self, predictions): 
        arm_cls, arm_loc, odm_cls, odm_loc = predictions
       
        arm_cls = self.predictions_transform(arm_cls, 2)
        arm_loc = self.predictions_transform(arm_loc, 4)
        odm_cls = self.predictions_transform(odm_cls, self._num_class)
        odm_loc = self.predictions_transform(odm_loc, 4)

        tf.identity(arm_cls, name='arm_cls')
        tf.identity(arm_loc, name='arm_loc')
        tf.identity(odm_cls, name='odm_cls')
        tf.identity(odm_loc, name='odm_loc')

        def decode_fn(arm_cls, arm_loc, odm_cls, odm_loc, priorboxes=self._priorboxes, variances=self._variances, num_class=self._num_class, config=self._detect_config):
            #arm_loc = transform_yx2xy(arm_loc)
            #odm_loc = transform_yx2xy(odm_loc)
            priorboxes_refine = to_center(decode(arm_loc, priorboxes, variances))
            decode_bboxes = decode(odm_loc, priorboxes_refine, variances) 
           
            arm_conf_pred = tf.nn.softmax(arm_cls)  
            odm_conf_pred = tf.nn.softmax(odm_cls)
     
            arm_filter_mask = arm_conf_pred[:, 1] > config.filter_obj_score
            arm_filter_mask = tf.matmul(tf.cast(tf.reshape(arm_filter_mask, shape=[-1, 1]), dtype=tf.float32), tf.ones(shape=[1, num_class])) 
            odm_conf_pred = tf.multiply(odm_conf_pred, arm_filter_mask)
            odm_conf_pred = tf.slice(odm_conf_pred, [0, 1], [tf.shape(odm_conf_pred)[0], num_class - 1])           
              
            detection_scores, detection_classes, detection_bboxes = detect_post_process(odm_conf_pred, decode_bboxes, self._num_class, config)  

            return detection_scores, detection_classes, detection_bboxes

        lamb_decode_fn = lambda x: decode_fn(x[0], x[1], x[2], x[3])
        det_results = tf.map_fn(lamb_decode_fn, (arm_cls, arm_loc, odm_cls, odm_loc), dtype=(tf.float32, tf.int64, tf.float32), back_prop=False)

        detection_scores = tf.concat(det_results[0], axis=0)
        detection_classes = tf.concat(det_results[1], axis=0)
        detection_bboxes = tf.concat(det_results[2], axis=0)

        return detection_scores, detection_classes, detection_bboxes 
