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


import numba
import pathlib
import numpy as np
import pickle
from functools import partial
from collections import defaultdict

from utils import box_np_ops
from datasets import kitti_common as kitti


def merge_second_batch(batch_list, _unused=False):
    example_merged = defaultdict(list)
    for example in batch_list:
        for k, v in example.items():
            example_merged[k].append(v)
    ret = {}
    example_merged.pop("num_voxels")
    for key, elems in example_merged.items():
        if key in [
                'voxels', 'num_points', 'num_gt', 'gt_boxes', 'voxel_labels',
                'match_indices'
        ]:  
            ret[key] = np.concatenate(elems, axis=0)
        elif key == 'match_indices_num':
            ret[key] = np.concatenate(elems, axis=0)
        elif key == 'coordinates':
            coors = []
            for i, coor in enumerate(elems):
                coor_pad = np.pad(
                    coor, ((0, 0), (1, 0)),
                    mode='constant',
                    constant_values=i)
                coors.append(coor_pad)
            ret[key] = np.concatenate(coors, axis=0)
        else:
            ret[key] = np.stack(elems, axis=0)
    return ret 


def _read_and_prep_v9(info, root_path, num_point_features, prep_func):
    """read data from KITTI-format infos, then call prep function.
    """
    # velodyne_path = str(pathlib.Path(root_path) / info['velodyne_path'])
    # velodyne_path += '_reduced'
    v_path = pathlib.Path(root_path) / info['velodyne_path']
    v_path = v_path.parent.parent / ( 
        v_path.parent.stem + "_reduced") / v_path.name

    points = np.fromfile(
        str(v_path), dtype=np.float32,
        count=-1).reshape([-1, num_point_features])
    image_idx = info['image_idx']
    rect = info['calib/R0_rect'].astype(np.float32)
    Trv2c = info['calib/Tr_velo_to_cam'].astype(np.float32)
    P2 = info['calib/P2'].astype(np.float32)

    input_dict = {
        'points': points,
        'rect': rect,
        'Trv2c': Trv2c,
        'P2': P2,
        'image_shape': np.array(info["img_shape"], dtype=np.int32),
        'image_idx': image_idx,
        'image_path': info['img_path'],
        # 'pointcloud_num_features': num_point_features,
    }

    if 'annos' in info:
        annos = info['annos']
        # we need other objects to avoid collision when sample
        annos = kitti.remove_dontcare(annos)
        loc = annos["location"]
        dims = annos["dimensions"]
        rots = annos["rotation_y"]
        gt_names = annos["name"]
        # print(gt_names, len(loc))
        gt_boxes = np.concatenate(
            [loc, dims, rots[..., np.newaxis]], axis=1).astype(np.float32)
        # gt_boxes = box_np_ops.box_camera_to_lidar(gt_boxes, rect, Trv2c)
        difficulty = annos["difficulty"]
        input_dict.update({
            'gt_boxes': gt_boxes,
            'gt_names': gt_names,
            'difficulty': difficulty,
        })
        if 'group_ids' in annos:
            input_dict['group_ids'] = annos["group_ids"]
    example = prep_func(input_dict=input_dict)
    example["image_idx"] = image_idx
    example["image_shape"] = input_dict["image_shape"]
    if "anchors_mask" in example:
        example["anchors_mask"] = example["anchors_mask"].astype(np.uint8)
    return example


class Dataset(object):
    """An abstract class representing a pytorch-like Dataset.
    All other datasets should subclass it. All subclasses should override
    ``__len__``, that provides the size of the dataset, and ``__getitem__``,
    supporting integer indexing in range from 0 to len(self) exclusive.
    """

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError



class KittiDataset(Dataset):
    def __init__(self, info_path, root_path, num_point_features,
                 target_assigner, feature_map_size, prep_func):
        with open(info_path, 'rb') as f:
            infos = pickle.load(f)
        #self._kitti_infos = kitti.filter_infos_by_used_classes(infos, class_names)
        self._root_path = root_path
        self._kitti_infos = infos
        self._num_point_features = num_point_features
        print("remain number of infos:", len(self._kitti_infos))
        # generate anchors cache
        # [352, 400]
        ret = target_assigner.generate_anchors(feature_map_size)
        anchors = ret["anchors"]
        anchors = anchors.reshape([-1, 7])
        matched_thresholds = ret["matched_thresholds"]
        unmatched_thresholds = ret["unmatched_thresholds"]
        anchors_bv = box_np_ops.rbbox2d_to_near_bbox(
            anchors[:, [0, 1, 3, 4, 6]])
        anchor_cache = {
            "anchors": anchors,
            "anchors_bv": anchors_bv,
            "matched_thresholds": matched_thresholds,
            "unmatched_thresholds": unmatched_thresholds,
        }
        self._prep_func = partial(prep_func, anchor_cache=anchor_cache)

    def __len__(self):
        return len(self._kitti_infos)

    @property
    def kitti_infos(self):
        return self._kitti_infos

    def __getitem__(self, idx):
        return _read_and_prep_v9(
            info=self._kitti_infos[idx],
            root_path=self._root_path,
            num_point_features=self._num_point_features,
            prep_func=self._prep_func)

@numba.jit(nopython=True)
def _points_to_bevmap_reverse_kernel(points,
                                     voxel_size,
                                     coors_range,
                                     coor_to_voxelidx,
                                     # coors_2d,
                                     bev_map,
                                     height_lowers,
                                     # density_norm_num=16,
                                     with_reflectivity=False,
                                     max_voxels=40000):
    # put all computations to one loop.
    # we shouldn't create large array in main jit code, otherwise
    # reduce performance
    N = points.shape[0]
    ndim = points.shape[1] - 1
    # ndim = 3
    ndim_minus_1 = ndim - 1
    grid_size = (coors_range[3:] - coors_range[:3]) / voxel_size
    # np.round(grid_size)
    # grid_size = np.round(grid_size).astype(np.int64)(np.int32)
    grid_size = np.round(grid_size, 0, grid_size).astype(np.int32)
    height_slice_size = voxel_size[-1]
    coor = np.zeros(shape=(3, ), dtype=np.int32)  # DHW
    voxel_num = 0
    failed = False
    for i in range(N):
        failed = False
        for j in range(ndim):
            c = np.floor((points[i, j] - coors_range[j]) / voxel_size[j])
            if c < 0 or c >= grid_size[j]:
                failed = True
                break
            coor[ndim_minus_1 - j] = c
        if failed:
            continue
        voxelidx = coor_to_voxelidx[coor[0], coor[1], coor[2]]
        if voxelidx == -1:
            voxelidx = voxel_num
            if voxel_num >= max_voxels:
                break
            voxel_num += 1
            coor_to_voxelidx[coor[0], coor[1], coor[2]] = voxelidx
            # coors_2d[voxelidx] = coor[1:]
        bev_map[-1, coor[1], coor[2]] += 1
        height_norm = bev_map[coor[0], coor[1], coor[2]]
        incomimg_height_norm = (
            points[i, 2] - height_lowers[coor[0]]) / height_slice_size
        if incomimg_height_norm > height_norm:
            bev_map[coor[0], coor[1], coor[2]] = incomimg_height_norm
            if with_reflectivity:
                bev_map[-2, coor[1], coor[2]] = points[i, 3]
    # return voxel_num


def points_to_bev(points,
                  voxel_size,
                  coors_range,
                  with_reflectivity=False,
                  density_norm_num=16,
                  max_voxels=40000):
    """convert kitti points(N, 4) to a bev map. return [C, H, W] map.
    this function based on algorithm in points_to_voxel.
    takes 5ms in a reduced pointcloud with voxel_size=[0.1, 0.1, 0.8]

    Args:
        points: [N, ndim] float tensor. points[:, :3] contain xyz points and
            points[:, 3] contain reflectivity.
        voxel_size: [3] list/tuple or array, float. xyz, indicate voxel size
        coors_range: [6] list/tuple or array, float. indicate voxel range.
            format: xyzxyz, minmax
        with_reflectivity: bool. if True, will add a intensity map to bev map.
    Returns:
        bev_map: [num_height_maps + 1(2), H, W] float tensor. 
            `WARNING`: bev_map[-1] is num_points map, NOT density map, 
            because calculate density map need more time in cpu rather than gpu. 
            if with_reflectivity is True, bev_map[-2] is intensity map. 
    """
    if not isinstance(voxel_size, np.ndarray):
        voxel_size = np.array(voxel_size, dtype=points.dtype)
    if not isinstance(coors_range, np.ndarray):
        coors_range = np.array(coors_range, dtype=points.dtype)
    voxelmap_shape = (coors_range[3:] - coors_range[:3]) / voxel_size
    voxelmap_shape = tuple(np.round(voxelmap_shape).astype(np.int32).tolist())
    voxelmap_shape = voxelmap_shape[::-1]  # DHW format
    coor_to_voxelidx = -np.ones(shape=voxelmap_shape, dtype=np.int32)
    # coors_2d = np.zeros(shape=(max_voxels, 2), dtype=np.int32)
    bev_map_shape = list(voxelmap_shape)
    bev_map_shape[0] += 1
    height_lowers = np.linspace(
        coors_range[2], coors_range[5], voxelmap_shape[0], endpoint=False)
    if with_reflectivity:
        bev_map_shape[0] += 1
    bev_map = np.zeros(shape=bev_map_shape, dtype=points.dtype)
    _points_to_bevmap_reverse_kernel(
        points, voxel_size, coors_range, coor_to_voxelidx, bev_map,
        height_lowers, with_reflectivity, max_voxels)
    return bev_map


def prep_pointcloud(input_dict,
                    root_path,
                    voxel_generator,
                    target_assigner,
                    db_sampler=None,
                    max_voxels=20000,
                    class_names=['Car'],
                    remove_outside_points=False,
                    training=True,
                    create_targets=True,
                    shuffle_points=False,
                    reduce_valid_area=False,
                    remove_unknown=False,
                    gt_rotation_noise=[-np.pi / 3, np.pi / 3], 
                    gt_loc_noise_std=[1.0, 1.0, 1.0],
                    global_rotation_noise=[-np.pi / 4, np.pi / 4], 
                    global_scaling_noise=[0.95, 1.05],
                    global_loc_noise_std=(0.2, 0.2, 0.2),
                    global_random_rot_range=[0.78, 2.35],
                    generate_bev=False,
                    without_reflectivity=False,
                    num_point_features=4,
                    anchor_area_threshold=1,
                    gt_points_drop=0.0,
                    gt_drop_max_keep=10,
                    remove_points_after_sample=True,
                    anchor_cache=None,
                    remove_environment=False,
                    random_crop=False,
                    reference_detections=None,
                    add_rgb_to_points=False,
                    lidar_input=False,
                    unlabeled_db_sampler=None,
                    out_size_factor=2,
                    min_gt_point_dict=None,
                    bev_only=False,
                    use_group_id=False,
                    out_dtype=np.float32):
    """convert point cloud to voxels, create targets if ground truths 
    exists.
    """
    points = input_dict["points"]
    rect = input_dict["rect"]
    Trv2c = input_dict["Trv2c"]
    P2 = input_dict["P2"]

    '''
    if reference_detections is not None:
        C, R, T = box_np_ops.projection_matrix_to_CRT_kitti(P2)
        frustums = box_np_ops.get_frustum_v2(reference_detections, C)
        frustums -= T
        # frustums = np.linalg.inv(R) @ frustums.T
        frustums = np.einsum('ij, akj->aki', np.linalg.inv(R), frustums)
        frustums = box_np_ops.camera_to_lidar(frustums, rect, Trv2c)
        surfaces = box_np_ops.corner_to_surfaces_3d_jit(frustums)
        masks = points_in_convex_polygon_3d_jit(points, surfaces)
        points = points[masks.any(-1)]

    if remove_outside_points and not lidar_input:
        image_shape = input_dict["image_shape"]
        points = box_np_ops.remove_outside_points(points, rect, Trv2c, P2,
                                                  image_shape)
    if remove_environment is True and training:
        selected = kitti.keep_arrays_by_name(gt_names, class_names)
        gt_boxes = gt_boxes[selected]
        gt_names = gt_names[selected]
        difficulty = difficulty[selected]
        if group_ids is not None:
            group_ids = group_ids[selected]
        points = prep.remove_points_outside_boxes(points, gt_boxes)
    '''

    if shuffle_points:
        # shuffle is a little slow.
        np.random.shuffle(points)

    # [0, -40, -3, 70.4, 40, 1]
    voxel_size = voxel_generator.voxel_size
    pc_range = voxel_generator.point_cloud_range
    grid_size = voxel_generator.grid_size
    # [352, 400]

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
    })
    # if not lidar_input:
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
        anchors_bv = box_np_ops.rbbox2d_to_near_bbox(
            anchors[:, [0, 1, 3, 4, 6]])
    example["anchors"] = anchors
    # print("debug", anchors.shape, matched_thresholds.shape)
    # anchors_bv = anchors_bv.reshape([-1, 4])
    anchors_mask = None
    if anchor_area_threshold >= 0:
        coors = coordinates
        dense_voxel_map = box_np_ops.sparse_sum_for_anchors_mask(
            coors, tuple(grid_size[::-1][1:]))
        dense_voxel_map = dense_voxel_map.cumsum(0)
        dense_voxel_map = dense_voxel_map.cumsum(1)
        anchors_area = box_np_ops.fused_get_anchors_area(
            dense_voxel_map, anchors_bv, voxel_size, pc_range, grid_size)
        anchors_mask = anchors_area > anchor_area_threshold
        # example['anchors_mask'] = anchors_mask.astype(np.uint8)
        example['anchors_mask'] = anchors_mask
    if generate_bev:
        bev_vxsize = voxel_size.copy()
        bev_vxsize[:2] /= 2
        bev_vxsize[2] *= 2
        bev_map = points_to_bev(points, bev_vxsize, pc_range,
                                without_reflectivity)
        example["bev_map"] = bev_map
    return example

