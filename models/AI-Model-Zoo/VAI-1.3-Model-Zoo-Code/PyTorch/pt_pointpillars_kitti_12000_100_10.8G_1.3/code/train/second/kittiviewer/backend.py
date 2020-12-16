
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

import fire 
import os
import numpy as np
import base64
import json
import time
from flask import Flask, jsonify, request
from flask_cors import CORS

import pickle
import sys
from functools import partial
from pathlib import Path

import second.core.box_np_ops as box_np_ops
import second.core.preprocess as prep
from second.core.box_coders import GroundBox3dCoder
from second.core.point_cloud.point_cloud_ops import points_to_voxel
from second.core.region_similarity import (
    DistanceSimilarity, NearestIouSimilarity, RotateIouSimilarity)
from second.core.sample_ops import DataBaseSamplerV2
from second.core.target_assigner import TargetAssigner
from second.data import kitti_common as kitti
from second.protos import pipeline_pb2
from second.utils.eval import get_coco_eval_result, get_official_eval_result
from second.pytorch.inference import TorchInferenceContext
from second.utils.progress_bar import list_bar

app = Flask("second")
CORS(app)

class SecondBackend:
    def __init__(self):
        self.root_path = None 
        self.info_path = None 
        self.kitti_infos = None
        self.image_idxes = None
        self.dt_annos = None
        self.inference_ctx = None


BACKEND = SecondBackend()

def error_response(msg):
    response = {}
    response["status"] = "error"
    response["message"] = "[ERROR]" + msg
    print("[ERROR]" + msg)
    return response


@app.route('/api/readinfo', methods=['POST'])
def readinfo():
    global BACKEND
    instance = request.json
    root_path = Path(instance["root_path"])
    
    
    response = {"status": "normal"}
    if not (root_path / "training").exists():
        response["status"] = "error"
        response["message"] = "ERROR: your root path is incorrect."
        print("ERROR: your root path is incorrect.")
        return response
    BACKEND.root_path = root_path
    info_path = Path(instance["info_path"])
    
    if not info_path.exists():
        response["status"] = "error"
        response["message"] = "ERROR: info file not exist."
        print("ERROR: your root path is incorrect.")
        return response
    BACKEND.info_path = info_path
    with open(info_path, 'rb') as f:
        kitti_infos = pickle.load(f)
    BACKEND.kitti_infos = kitti_infos
    BACKEND.image_idxes = [info["image_idx"] for info in kitti_infos]
    response["image_indexes"] = BACKEND.image_idxes

    response = jsonify(results=[response])
    response.headers['Access-Control-Allow-Headers'] = '*'
    return response

@app.route('/api/read_detection', methods=['POST'])
def read_detection():
    global BACKEND
    instance = request.json
    det_path = Path(instance["det_path"])
    response = {"status": "normal"}
    if BACKEND.root_path is None:
        return error_response("root path is not set")
    if BACKEND.kitti_infos is None:
        return error_response("kitti info is not loaded")

    if Path(det_path).is_file():
        with open(det_path, "rb") as f:
            dt_annos = pickle.load(f)
    else:
        dt_annos = kitti.get_label_annos(det_path)
    BACKEND.dt_annos = dt_annos
    response = jsonify(results=[response])
    response.headers['Access-Control-Allow-Headers'] = '*'
    return response


@app.route('/api/get_pointcloud', methods=['POST'])
def get_pointcloud():
    global BACKEND
    instance = request.json
    response = {"status": "normal"}
    if BACKEND.root_path is None:
        return error_response("root path is not set")
    if BACKEND.kitti_infos is None:
        return error_response("kitti info is not loaded")
    image_idx = instance["image_idx"]
    idx = BACKEND.image_idxes.index(image_idx)
    kitti_info = BACKEND.kitti_infos[idx]
    rect = kitti_info['calib/R0_rect']
    P2 = kitti_info['calib/P2']
    Trv2c = kitti_info['calib/Tr_velo_to_cam']
    img_shape = kitti_info["img_shape"] # hw
    wh = np.array(img_shape[::-1])
    whwh = np.tile(wh, 2)
    if 'annos' in kitti_info:
        annos = kitti_info['annos']
        labels = annos['name']
        num_obj = len([n for n in annos['name'] if n != 'DontCare'])
        dims = annos['dimensions'][:num_obj]
        loc = annos['location'][:num_obj]
        rots = annos['rotation_y'][:num_obj]
        bbox = annos['bbox'][:num_obj] / whwh
        gt_boxes_camera = np.concatenate(
            [loc, dims, rots[..., np.newaxis]], axis=1)
        gt_boxes = box_np_ops.box_camera_to_lidar(
            gt_boxes_camera, rect, Trv2c)
        box_np_ops.change_box3d_center_(gt_boxes, src=[0.5, 0.5, 0], dst=[0.5, 0.5, 0.5])
        locs = gt_boxes[:, :3]
        dims = gt_boxes[:, 3:6]
        rots = np.concatenate([np.zeros([num_obj, 2], dtype=np.float32), -gt_boxes[:, 6:7]], axis=1)
        frontend_annos = {}
        response["locs"] = locs.tolist()
        response["dims"] = dims.tolist()
        response["rots"] = rots.tolist()
        response["bbox"] = bbox.tolist()
        
        response["labels"] = labels[:num_obj].tolist()

    v_path = str(Path(BACKEND.root_path) / kitti_info['velodyne_path'])
    with open(v_path, 'rb') as f:
        pc_str = base64.encodestring(f.read())
    response["pointcloud"] = pc_str.decode("utf-8")
    if "with_det" in instance and instance["with_det"]:
        if BACKEND.dt_annos is None:
            return error_response("det anno is not loaded")
        dt_annos = BACKEND.dt_annos[idx]
        dims = dt_annos['dimensions']
        num_obj = dims.shape[0]
        loc = dt_annos['location']
        rots = dt_annos['rotation_y']
        bbox = dt_annos['bbox'] / whwh
        labels = dt_annos['name']

        dt_boxes_camera = np.concatenate(
            [loc, dims, rots[..., np.newaxis]], axis=1)
        dt_boxes = box_np_ops.box_camera_to_lidar(
            dt_boxes_camera, rect, Trv2c)
        box_np_ops.change_box3d_center_(dt_boxes, src=[0.5, 0.5, 0], dst=[0.5, 0.5, 0.5])
        locs = dt_boxes[:, :3]
        dims = dt_boxes[:, 3:6]
        rots = np.concatenate([np.zeros([num_obj, 2], dtype=np.float32), -dt_boxes[:, 6:7]], axis=1)
        response["dt_locs"] = locs.tolist()
        response["dt_dims"] = dims.tolist()
        response["dt_rots"] = rots.tolist()
        response["dt_labels"] = labels.tolist()
        response["dt_bbox"] = bbox.tolist()
        response["dt_scores"] = dt_annos["score"].tolist()

    # if "score" in annos:
    #     response["score"] = score.tolist()
    response = jsonify(results=[response])
    response.headers['Access-Control-Allow-Headers'] = '*'
    print("send response!")
    return response

@app.route('/api/get_image', methods=['POST'])
def get_image():
    global BACKEND
    instance = request.json
    response = {"status": "normal"}
    if BACKEND.root_path is None:
        return error_response("root path is not set")
    if BACKEND.kitti_infos is None:
        return error_response("kitti info is not loaded")
    image_idx = instance["image_idx"]
    idx = BACKEND.image_idxes.index(image_idx)
    kitti_info = BACKEND.kitti_infos[idx]
    rect = kitti_info['calib/R0_rect']
    P2 = kitti_info['calib/P2']
    Trv2c = kitti_info['calib/Tr_velo_to_cam']
    if 'img_path' in kitti_info:
        img_path = kitti_info['img_path']
        if img_path != "":
            image_path = BACKEND.root_path / img_path
            print(image_path)
            with open(str(image_path), 'rb') as f:
                image_str = f.read()
            response["image_b64"] = base64.b64encode(image_str).decode("utf-8")
            response["image_b64"] = 'data:image/{};base64,'.format(image_path.suffix[1:]) + response["image_b64"]
            '''# 
            response["rect"] = rect.tolist()
            response["P2"] = P2.tolist()
            response["Trv2c"] = Trv2c.tolist()
            response["L2CMat"] = ((rect @ Trv2c).T).tolist()
            response["C2LMat"] = np.linalg.inv((rect @ Trv2c).T).tolist()
            '''
            print("send an image with size {}!".format(len(response["image_b64"])))
    response = jsonify(results=[response])
    response.headers['Access-Control-Allow-Headers'] = '*'
    return response

@app.route('/api/build_network', methods=['POST'])
def build_network():
    global BACKEND
    instance = request.json
    cfg_path = Path(instance["config_path"])
    ckpt_path = Path(instance["checkpoint_path"])
    response = {"status": "normal"}
    if BACKEND.root_path is None:
        return error_response("root path is not set")
    if BACKEND.kitti_infos is None:
        return error_response("kitti info is not loaded")
    if not cfg_path.exists():
        return error_response("config file not exist.")
    if not ckpt_path.exists():
        return error_response("ckpt file not exist.")
    BACKEND.inference_ctx = TorchInferenceContext()
    BACKEND.inference_ctx.build(str(cfg_path))
    BACKEND.inference_ctx.restore(str(ckpt_path))
    response = jsonify(results=[response])
    response.headers['Access-Control-Allow-Headers'] = '*'
    print("build_network successful!")
    return response


@app.route('/api/inference_by_idx', methods=['POST'])
def inference_by_idx():
    global BACKEND
    instance = request.json
    response = {"status": "normal"}
    if BACKEND.root_path is None:
        return error_response("root path is not set")
    if BACKEND.kitti_infos is None:
        return error_response("kitti info is not loaded")
    if BACKEND.inference_ctx is None:
        return error_response("inference_ctx is not loaded")
    image_idx = instance["image_idx"]
    idx = BACKEND.image_idxes.index(image_idx)
    kitti_info = BACKEND.kitti_infos[idx]

    v_path = str(Path(BACKEND.root_path) / kitti_info['velodyne_path'])
    num_features = 4
    points = np.fromfile(
        str(v_path), dtype=np.float32,
        count=-1).reshape([-1, num_features])
    rect = kitti_info['calib/R0_rect']
    P2 = kitti_info['calib/P2']
    Trv2c = kitti_info['calib/Tr_velo_to_cam']
    if 'img_shape' in kitti_info:
        image_shape = kitti_info['img_shape']
        points = box_np_ops.remove_outside_points(
            points, rect, Trv2c, P2, image_shape)
        print(points.shape[0])
    img_shape = kitti_info["img_shape"] # hw
    wh = np.array(img_shape[::-1])
    whwh = np.tile(wh, 2)

    t = time.time()
    inputs = BACKEND.inference_ctx.get_inference_input_dict(
        kitti_info, points)
    print("input preparation time:", time.time() - t)
    t = time.time()
    with BACKEND.inference_ctx.ctx():
        dt_annos = BACKEND.inference_ctx.inference(inputs)[0]
    print("detection time:", time.time() - t)
    dims = dt_annos['dimensions']
    num_obj = dims.shape[0]
    loc = dt_annos['location']
    rots = dt_annos['rotation_y']
    labels = dt_annos['name']
    bbox = dt_annos['bbox'] / whwh

    dt_boxes_camera = np.concatenate(
        [loc, dims, rots[..., np.newaxis]], axis=1)
    dt_boxes = box_np_ops.box_camera_to_lidar(
        dt_boxes_camera, rect, Trv2c)
    box_np_ops.change_box3d_center_(dt_boxes, src=[0.5, 0.5, 0], dst=[0.5, 0.5, 0.5])
    locs = dt_boxes[:, :3]
    dims = dt_boxes[:, 3:6]
    rots = np.concatenate([np.zeros([num_obj, 2], dtype=np.float32), -dt_boxes[:, 6:7]], axis=1)
    response["dt_locs"] = locs.tolist()
    response["dt_dims"] = dims.tolist()
    response["dt_rots"] = rots.tolist()
    response["dt_labels"] = labels.tolist()
    response["dt_scores"] = dt_annos["score"].tolist()
    response["dt_bbox"] = bbox.tolist()

    response = jsonify(results=[response])
    response.headers['Access-Control-Allow-Headers'] = '*'
    return response


def main(port=16666):
    app.run(host='0.0.0.0', threaded=True, port=port)

if __name__ == '__main__':
    fire.Fire()
