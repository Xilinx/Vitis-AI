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


# This code is based on: https://github.com/nutonomy/second.pytorch.git
# 
# MIT License
# Copyright (c) 2018 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import warnings
warnings.filterwarnings("ignore")

import torch
import time
import os
import cv2
import argparse
import numpy as np
from pathlib import Path
import pickle
from easydict import EasyDict
from builder import voxel_builder
from builder import box_coder_builder
from builder import target_assigner_builder
from builder import second_builder
from builder import input_reader_builder
from datasets.dataset_tools import merge_second_batch
import datasets.kitti_common as kitti
from utils.progress_bar import ProgressBar
from utils.eval import get_official_eval_result, get_coco_eval_result
from utils.viz_tool import preprocess, label2BevImage_v4, drawRotBboxes, get_2DBboxes_from_anno, draw2DBboxes, draw_proj_box, anno2objs
from utils import box_np_ops
from second.protos import pipeline_pb2
from google.protobuf import text_format
from tqdm import tqdm

from datasets.kitti_util import Calibration

if os.environ["W_QUANT"]=='1':
    from pytorch_nndct.apis import torch_quantizer


def prep_input_dict(points, rect, Trv2c, P2, image_shape, image_idx):
    input_dict = {
        'points': points,
        'rect': rect,
        'Trv2c': Trv2c,
        'P2': P2,
        'image_shape': np.array(image_shape, dtype=np.int32),
        'image_idx': image_idx,
    }
    return input_dict


def prep_pointcloud(input_dict,
                    voxel_generator,
                    target_assigner,
                    max_voxels,
                    out_size_factor=2,
                    anchor_cache=None,
                    anchor_area_threshold=1,
                    shuffle_points=False):
    """convert point cloud to voxels, create targets if ground truths exists.
    """
    points = input_dict["points"]
    rect = input_dict["rect"]
    Trv2c = input_dict["Trv2c"]
    P2 = input_dict["P2"]
    image_shape = input_dict["image_shape"]
    image_idx = input_dict['image_idx']

    if shuffle_points:
        np.random.shuffle(points)

    # [0, -40, -3, 70.4, 40, 1]
    voxel_size = voxel_generator.voxel_size
    pc_range = voxel_generator.point_cloud_range
    grid_size = voxel_generator.grid_size

    voxels, coordinates, num_points = voxel_generator.generate(points, max_voxels)

    example = {
        'voxels': voxels,
        'num_points': num_points,
        'coordinates': coordinates,
        "num_voxels": np.array([voxels.shape[0]], dtype=np.int64)
    }
    example.update({
        'rect': rect,
        'Trv2c': Trv2c,
        'P2': P2,
        'image_shape': image_shape,
        'image_idx': image_idx,
    })

    feature_map_size = grid_size[:2] // out_size_factor
    feature_map_size = [*feature_map_size, 1][::-1]
    if anchor_cache is not None:
        anchors = anchor_cache["anchors"]
        anchors_bv = anchor_cache["anchors_bv"]
        matched_thresholds = anchor_cache["matched_thresholds"]
        unmatched_thresholds = anchor_cache["unmatched_thresholds"]
    else:
        ret = target_assigner.generate_anchors(feature_map_size)
        anchors = ret["anchors"]
        anchors = anchors.reshape([-1, 7])
        matched_thresholds = ret["matched_thresholds"]
        unmatched_thresholds = ret["unmatched_thresholds"]
        anchors_bv = box_np_ops.rbbox2d_to_near_bbox(anchors[:, [0, 1, 3, 4, 6]])
    example["anchors"] = anchors
    # print("debug", anchors.shape, matched_thresholds.shape)
    # anchors_bv = anchors_bv.reshape([-1, 4])
    anchors_mask = None
    if anchor_area_threshold >= 0:
        coors = coordinates
        dense_voxel_map = box_np_ops.sparse_sum_for_anchors_mask(coors, tuple(grid_size[::-1][1:]))
        dense_voxel_map = dense_voxel_map.cumsum(0)
        dense_voxel_map = dense_voxel_map.cumsum(1)
        anchors_area = box_np_ops.fused_get_anchors_area(dense_voxel_map, anchors_bv, voxel_size, pc_range, grid_size)
        anchors_mask = anchors_area > anchor_area_threshold
        # example['anchors_mask'] = anchors_mask.astype(np.uint8)
        example['anchors_mask'] = anchors_mask
    return example


class KITTIRawData(object):

    def __init__(self, root, rect, Trv2c, P2, image_shape):
        self.img_root = os.path.join(root, 'image_02', 'data')
        self.velo_root = os.path.join(root, 'velodyne_points', 'data')
        self.img_names = sorted(os.listdir(self.img_root))
        self.velo_names = sorted(os.listdir(self.velo_root))
        assert len(self.img_names) == len(self.velo_names)
        self.rect = rect
        self.Trv2c = Trv2c
        self.P2 = P2
        self.image_shape = image_shape

    def __len__(self):
        return len(self.velo_names)

    def __getitem__(self, idx):
        velo_path = os.path.join(self.velo_root, self.velo_names[idx])
        rgb_path = os.path.join(self.img_root, self.img_names[idx])
        points = np.fromfile(velo_path, dtype=np.float32, count=-1).reshape([-1, 4])
        input_dict = prep_input_dict(points, self.rect, self.Trv2c, self.P2, self.image_shape, idx)
        return input_dict, velo_path, rgb_path


def example_convert_to_torch(example, dtype=torch.float32,
                             device=None) -> dict:
    device = device or torch.device("cuda:0")
    example_torch = {}
    float_names = [ 
        "voxels", "anchors", "reg_targets", "reg_weights", "bev_map", "rect",
        "Trv2c", "P2"
    ]   

    for k, v in example.items():
        if k in float_names:
            example_torch[k] = torch.as_tensor(v, dtype=dtype, device=device)
        elif k in ["coordinates", "labels", "num_points"]:
            example_torch[k] = torch.as_tensor(
                v, dtype=torch.int32, device=device)
        elif k in ["anchors_mask"]:
            example_torch[k] = torch.as_tensor(
                v, dtype=torch.uint8, device=device)
        else:
            example_torch[k] = v 
    return example_torch


def predict_kitti_to_anno(net,
                          preprocessor,
                          postprocessor,
                          example,
                          class_names,
                          center_limit_range=None,
                          lidar_input=False,
                          global_set=None):
    batch_image_shape = example['image_shape']
    voxels = example["voxels"]
    num_points = example["num_points"]
    coors = example["coordinates"]

    aug_voxels = preprocessor(voxels, num_points, coors)
    pred_tuples = net.forward(aug_voxels, coors)

    preds_dict = {}
    if len(pred_tuples) == 3:
        box_preds, cls_preds, dir_cls_preds = pred_tuples
        preds_dict['box_preds'] = box_preds
        preds_dict['cls_preds'] = cls_preds
        preds_dict['dir_cls_preds'] = dir_cls_preds
    else:
        box_preds, cls_preds = pred_tuples
        preds_dict['box_preds'] = box_preds
        preds_dict['cls_preds'] = cls_preds

    predictions_dicts = postprocessor(example, preds_dict)

    annos = []
    for i, preds_dict in enumerate(predictions_dicts):
        image_shape = batch_image_shape[i]
        img_idx = preds_dict["image_idx"]
        if preds_dict["bbox"] is not None:
            box_2d_preds = preds_dict["bbox"].detach().cpu().numpy()
            box_preds = preds_dict["box3d_camera"].detach().cpu().numpy()
            scores = preds_dict["scores"].detach().cpu().numpy()
            box_preds_lidar = preds_dict["box3d_lidar"].detach().cpu().numpy()
            # write pred to file
            label_preds = preds_dict["label_preds"].detach().cpu().numpy()
            # label_preds = np.zeros([box_2d_preds.shape[0]], dtype=np.int32)
            # anno = kitti.get_start_result_anno()
            anno = kitti.get_demo_anno()
            num_example = 0
            for box, box_lidar, bbox, score, label in zip(
                    box_preds, box_preds_lidar, box_2d_preds, scores,
                    label_preds):
                if not lidar_input:
                    if bbox[0] > image_shape[1] or bbox[1] > image_shape[0]:
                        continue
                    if bbox[2] < 0 or bbox[3] < 0:
                        continue
                if center_limit_range is not None:
                    limit_range = np.array(center_limit_range)
                    if (np.any(box_lidar[:3] < limit_range[:3])
                            or np.any(box_lidar[:3] > limit_range[3:])):
                        continue
                bbox[2:] = np.minimum(bbox[2:], image_shape[::-1])
                bbox[:2] = np.maximum(bbox[:2], [0, 0])
                anno["name"].append(class_names[int(label)])
                anno["truncated"].append(0.0)
                anno["occluded"].append(0)
                anno["alpha"].append(-np.arctan2(-box_lidar[1], box_lidar[0]) +
                                     box[6])
                anno["bbox"].append(bbox)
                anno["dimensions"].append(box[3:6])
                anno["location"].append(box[:3])
                anno["rotation_y"].append(box[6])
                anno['box3d_camera'].append(box)
                anno['box3d_lidar'].append(box_lidar)
                if global_set is not None:
                    for i in range(100000):
                        if score in global_set:
                            score -= 1 / 100000
                        else:
                            global_set.add(score)
                            break
                anno["score"].append(score)

                num_example += 1
            if num_example != 0:
                anno = {n: np.stack(v) for n, v in anno.items()}
                annos.append(anno)
            else:
                annos.append(kitti.empty_result_anno())
        else:
            annos.append(kitti.empty_result_anno())
        num_example = annos[-1]["name"].shape[0]
        annos[-1]["image_idx"] = np.array(
            [img_idx] * num_example, dtype=np.int64)
    return annos


def predict_kitti_to_file(net,
                          preprocessor,
                          postprocessor,
                          example,
                          result_dir,
                          class_names,
                          center_limit_range=None,
                          lidar_input=False):
    batch_image_shape = example['image_shape']
    batch_imgidx = example['image_idx']
    voxels = example["voxels"]
    num_points = example["num_points"]
    coors = example["coordinates"]

    aug_voxels = preprocessor(voxels, num_points, coors)
    pred_tuples = net.forward(aug_voxels, coors)

    preds_dict = {}
    if len(pred_tuples) == 3:
        box_preds, cls_preds, dir_cls_preds = pred_tuples
        preds_dict['box_preds'] = box_preds
        preds_dict['cls_preds'] = cls_preds
        preds_dict['dir_cls_preds'] = dir_cls_preds
    else:
        box_preds, cls_preds = pred_tuples
        preds_dict['box_preds'] = box_preds
        preds_dict['cls_preds'] = cls_preds

    predictions_dicts = postprocessor(example, preds_dict)

    # t = time.time()
    for i, preds_dict in enumerate(predictions_dicts):
        image_shape = batch_image_shape[i]
        img_idx = preds_dict["image_idx"]
        if preds_dict["bbox"] is not None:
            box_2d_preds = preds_dict["bbox"].data.cpu().numpy()
            box_preds = preds_dict["box3d_camera"].data.cpu().numpy()
            scores = preds_dict["scores"].data.cpu().numpy()
            box_preds_lidar = preds_dict["box3d_lidar"].data.cpu().numpy()
            # write pred to file
            box_preds = box_preds[:, [0, 1, 2, 4, 5, 3,
                                      6]]  # lhw->hwl(label file format)
            label_preds = preds_dict["label_preds"].data.cpu().numpy()
            # label_preds = np.zeros([box_2d_preds.shape[0]], dtype=np.int32)
            result_lines = []
            for box, box_lidar, bbox, score, label in zip(
                    box_preds, box_preds_lidar, box_2d_preds, scores,
                    label_preds):
                if not lidar_input:
                    if bbox[0] > image_shape[1] or bbox[1] > image_shape[0]:
                        continue
                    if bbox[2] < 0 or bbox[3] < 0:
                        continue
                # print(img_shape)
                if center_limit_range is not None:
                    limit_range = np.array(center_limit_range)
                    if (np.any(box_lidar[:3] < limit_range[:3])
                            or np.any(box_lidar[:3] > limit_range[3:])):
                        continue
                bbox[2:] = np.minimum(bbox[2:], image_shape[::-1])
                bbox[:2] = np.maximum(bbox[:2], [0, 0])
                result_dict = {
                    'name': class_names[int(label)],
                    'alpha': -np.arctan2(-box_lidar[1], box_lidar[0]) + box[6],
                    'bbox': bbox,
                    'location': box[:3],
                    'dimensions': box[3:6],
                    'rotation_y': box[6],
                    'score': score,
                }
                result_line = kitti.kitti_result_line(result_dict)
                result_lines.append(result_line)
        else:
            result_lines = []
        result_file = f"{result_dir}/{kitti.get_image_index_str(img_idx)}.txt"
        result_str = '\n'.join(result_lines)
        with open(result_file, 'w') as f:
            f.write(result_str)


def restore(ckpt_path, model):
    if not Path(ckpt_path).is_file():
        raise ValueError("checkpoint {} not exist.".format(ckpt_path))
    model_dict = model.state_dict()
    ckpt_dict = torch.load(ckpt_path)
    for k in model_dict.keys():
        model_dict[k] = ckpt_dict[k]
    model.load_state_dict(model_dict)
    print("Restoring parameters from {}".format(ckpt_path))


def generate_video(rgb_video, rgb3d_video, bev_video, dataset, config_path, ckpt_path, quant_mode, quant_dir, output_dir):

    config = pipeline_pb2.TrainEvalPipelineConfig()
    with open(config_path, "r") as f:
        proto_str = f.read()
        text_format.Merge(proto_str, config)

    input_cfg = config.eval_input_reader
    model_cfg = config.model.second
    train_cfg = config.train_config
    class_names = list(input_cfg.class_names)
    center_limit_range = model_cfg.post_center_limit_range

    voxel_generator = voxel_builder.build(model_cfg.voxel_generator)
    bv_range = voxel_generator.point_cloud_range[[0, 1, 3, 4]] 
    box_coder = box_coder_builder.build(model_cfg.box_coder)
    target_assigner_cfg = model_cfg.target_assigner
    target_assigner = target_assigner_builder.build(target_assigner_cfg, bv_range, box_coder)
    
    net, preprocessor, postprocessor = second_builder.build(model_cfg, voxel_generator, target_assigner)

    net.cuda()
    restore(ckpt_path, net)

    max_voxel_num = config.eval_input_reader.max_number_of_voxels
    max_point_num_per_voxel = model_cfg.voxel_generator.max_number_of_points_per_voxel
    if quant_mode != 'float':
        aug_voxels = torch.randn((1, 4, max_voxel_num, max_point_num_per_voxel)).cuda()
        coors = torch.randn((max_voxel_num, 4)).cuda()
        quantizer = torch_quantizer(quant_mode=quant_mode,
                                    module=net, 
                                    input_args=(aug_voxels, coors), 
                                    output_dir=quant_dir,
                                    bitwidth_w=8,
                                    bitwidth_a=8)
        net = quantizer.quant_model.cuda()
    net.eval()

    for input_dict, velo_path, rgb_path in tqdm(dataset):
        example = prep_pointcloud(input_dict, voxel_generator, target_assigner, max_voxel_num)
        example =  merge_second_batch([example])
        example = example_convert_to_torch(example, torch.float32)

        # predict_kitti_to_file(net, preprocessor, postprocessor, example, output_dir, class_names, center_limit_range, model_cfg.lidar_input)
        dt_anno = predict_kitti_to_anno(net, preprocessor, postprocessor, example, class_names, center_limit_range, model_cfg.lidar_input)

        dt_anno = dt_anno[0]
        P2 = input_dict['P2']
        rect = input_dict['rect']
        P2_rect = np.dot(P2, rect)

        bev_map, rgb_map, rgb_3d_map = get_image_with_prediction(velo_path, rgb_path, dt_anno, args.score_thresh, args.quant_dir, P2_rect)

        rgb_video.write(rgb_map)
        rgb3d_video.write(rgb_3d_map)
        bev_video.write(bev_map)


def get_image_with_prediction(lidar_path, image_path, dt_anno, score_thresh, result_dir, P2_rect,
                              boundaryCond = {'minX':0, 'maxX': 80, 'minY': -40, 'maxY': 40, 'minZ': -3, 'maxZ': 3}, 
                              resolution = 40./512):

    pointCloud = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 4)
    bev_map = preprocess(pointCloud, boundaryCond, resolution)
    bboxes = label2BevImage_v4(dt_anno, boundaryCond, resolution, score_thresh)
    bev_map = drawRotBboxes(bev_map, bboxes)
    bev_map = cv2.flip(bev_map, -1)

    rgb_map = cv2.imread(image_path)
    rgb_3d_map = rgb_map.copy()

    bboxes = get_2DBboxes_from_anno(dt_anno, score_thresh)
    rgb_map = draw2DBboxes(rgb_map, bboxes)

    objs = anno2objs(dt_anno, score_thresh)
    for obj in objs:
        rgb_3d_map = draw_proj_box(rgb_3d_map, obj, P2_rect)

    return bev_map, rgb_map, rgb_3d_map


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Demo script for PointPillars on KITTI dataset.')
    parser.add_argument('-config', default='float/config.proto')
    parser.add_argument('-weights', default='float/float.tckpt')
    parser.add_argument('-result_dir', default='data/result')
    parser.add_argument('-quant_dir', default='quantized')
    parser.add_argument('-quant_mode', default='test', choices=['float', 'test'], help='quantization mode. float: no quantization, evaluate float model, test: evaluate quantized model')
    parser.add_argument('-score_thresh', type=float, default=0.5)
    parser.add_argument('-data_root', default='data/KITTI/KITTI_Raw_Data/2011_09_26/2011_09_26_drive_0093_sync')
    parser.add_argument('-calib_dir', default='data/KITTI/KITTI_Raw_Data/2011_09_26')
    parser.add_argument('-save_prefix', type=str)
    parser.add_argument('-fps', type=int, default=10)
    args = parser.parse_args()

    fps = args.fps
    root = args.data_root
    prefix = args.save_prefix
    calib_dir = args.calib_dir

    calib = Calibration(calib_dir, from_video=True)
    P, R0, V2C = calib.P, calib.R0, calib.V2C
    rect = np.eye(4)
    rect[:3,:3] = R0
    P2 = np.eye(4)
    P2[:3,:4] = P
    Trv2c = np.eye(4)
    Trv2c[:3,:4] = V2C

    image_shape = [375, 1242]
    bev_shape = [512, 1024]

    dataset = KITTIRawData(root, rect, Trv2c, P2, image_shape)
    
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    rgb_video = cv2.VideoWriter(os.path.join(args.result_dir, prefix + '_rgb_demo.mp4'), fourcc, fps, (image_shape[1], image_shape[0]), True)
    rgb3d_video = cv2.VideoWriter(os.path.join(args.result_dir, prefix + '_rgb3d_demo.mp4'), fourcc, fps, (image_shape[1], image_shape[0]), True)
    bev_video = cv2.VideoWriter(os.path.join(args.result_dir, prefix + '_bev_demo.mp4'), fourcc, fps, (bev_shape[1], bev_shape[0]), True)

    generate_video(rgb_video, rgb3d_video, bev_video, dataset, args.config, args.weights, args.quant_mode, args.quant_dir, args.result_dir)
