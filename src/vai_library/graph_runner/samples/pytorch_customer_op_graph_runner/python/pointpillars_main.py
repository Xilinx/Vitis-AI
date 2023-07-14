"""
Copyright 2022-2023 Advanced Micro Devices Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from typing import List
import cv2
import numpy as np
import sys
import math
import xir
import vitis_ai_library

import struct

# 1. cfg items
point_cloud_range = [0, -39.68, -3, 69.12, 39.68, 1]
voxel_size = [0.16, 0.16, 4]
max_voxels = 12000
max_points = 100

anchor_sizes =[ [1.6, 3.9, 1.56 ],     [0.6, 1.76, 1.73] ,     [0.6, 0.8, 1.73]  ]
anchor_strides =[0.32, 0.32, 0.0]
anchor_offsets =[ [0.16, -39.52, -1.78], [0.16, -39.52, -1.465], [0.16, -39.52, -1.465]]
anchor_rotations= [0, 1.57]
matched_threshold = [0.6, 0.5, 0.5]
unmatched_threshold=[0.45, 0.35, 0.35]
out_size_factor = 2
anchor_area_threshold=1
nms_confidence = 0.5
nms_iou_threshold = 0.5

# 2. global var
global grid_size
global g_anchors
global g_anchors_bv

BOX_IDX=2
CLS_IDX=1
DIR_IDX=0

# 3. helper function

# 3.1
def get_grid_size():
  grid_size1=[]
  for i in range(3):
    grid_size1.append(round(( point_cloud_range[i+3]-point_cloud_range[i] )/voxel_size[i]))
  return grid_size1

# 3.2 anchor related

def limit_period(val, offset=0.5, period=np.pi):
    return val - np.floor(val / period + offset) * period

def center_to_minmax_2d_0_5(centers, dims):
    return np.concatenate([centers - dims / 2, centers + dims / 2], axis=-1)

def corners_nd(dims, origin=0.5):
    ndim = int(dims.shape[1])
    corners_norm = np.stack(
        np.unravel_index(np.arange(2**ndim), [2] * ndim), axis=1).astype(
            dims.dtype)
    #print("corners norm  ", corners_norm )
    # now corners_norm has format: (2d) x0y0, x0y1, x1y0, x1y1
    # (3d) x0y0z0, x0y0z1, x0y1z0, x0y1z1, x1y0z0, x1y0z1, x1y1z0, x1y1z1
    # so need to convert to a format which is convenient to do other computing.
    # for 2d boxes, format is clockwise start with minimum point
    # for 3d boxes, please draw lines by your hand.
    if ndim == 2:
        # generate clockwise box corners
        corners_norm = corners_norm[[0, 1, 3, 2]]
    elif ndim == 3:
        corners_norm = corners_norm[[0, 1, 3, 2, 4, 5, 7, 6]]
    corners_norm = corners_norm - np.array(origin, dtype=dims.dtype)
    corners = dims.reshape([-1, 1, ndim]) * corners_norm.reshape(
        [1, 2**ndim, ndim])
    return corners

def rotation_2d(points, angles):
    rot_sin = np.sin(angles)
    rot_cos = np.cos(angles)
    rot_mat_T = np.stack([[rot_cos, -rot_sin], [rot_sin, rot_cos]])
    return np.einsum('aij,jka->aik', points, rot_mat_T)

def center_to_corner_box2d(centers, dims, angles=None, origin=0.5):
    corners = corners_nd(dims, origin=origin)
    # corners: [N, 4, 2]
    if angles is not None:
        corners = rotation_2d(corners, angles)
    corners += centers.reshape([-1, 1, 2])
    return corners

def center_to_minmax_2d(centers, dims, origin=0.5):
    if origin == 0.5:
        return center_to_minmax_2d_0_5(centers, dims)
    corners = center_to_corner_box2d(centers, dims, origin=origin)
    return corners[:, [0, 2]].reshape([-1, 4])


def rbbox2d_to_near_bbox(rbboxes):
    """convert rotated bbox to nearest 'standing' or 'lying' bbox.
    Args:
        rbboxes: [N, 5(x, y, xdim, ydim, rad)] rotated bboxes
    Returns:
        bboxes: [N, 4(xmin, ymin, xmax, ymax)] bboxes
    """
    rots = rbboxes[..., -1]
    rots_0_pi_div_2 = np.abs(limit_period(rots, 0.5, np.pi))
    cond = (rots_0_pi_div_2 > np.pi / 4)[..., np.newaxis]
    bboxes_center = np.where(cond, rbboxes[:, [0, 1, 3, 2]], rbboxes[:, :4])
    bboxes = center_to_minmax_2d(bboxes_center[:, :2], bboxes_center[:, 2:])
    return bboxes


def create_anchor(grid_size):
    # print("anchor")
    feature_map_size = [ x// out_size_factor for x in grid_size[:2]]
    # feature_map_size = grid_size[:2] // out_size_factor
    feature_map_size = [*feature_map_size, 1][::-1]
    # print("feat " , feature_map_size )
    anchors_list = []

    for i in range(3):
       x_stride, y_stride, z_stride = anchor_strides
       # print("xya :" , x_stride, y_stride, z_stride  )
       x_offset, y_offset, z_offset = anchor_offsets[i]
       # print("offset : " , x_offset, y_offset, z_offset  )
       z_centers = np.arange(feature_map_size[0], dtype=np.float32)
       y_centers = np.arange(feature_map_size[1], dtype=np.float32)
       x_centers = np.arange(feature_map_size[2], dtype=np.float32)

       z_centers = z_centers * z_stride + z_offset
       y_centers = y_centers * y_stride + y_offset
       x_centers = x_centers * x_stride + x_offset
       # print("center : " , z_centers, y_centers, x_centers )

       sizes = np.reshape(np.array(anchor_sizes[i], dtype=np.float32), [-1, 3])
       rotations = np.array(anchor_rotations, dtype=np.float32)
       rets = np.meshgrid( x_centers, y_centers, z_centers, rotations, indexing='ij')
       tile_shape = [1] * 5
       tile_shape[-2] = int(sizes.shape[0])
       for k in range(len(rets)):
           rets[k] = np.tile(rets[k][..., np.newaxis, :], tile_shape)
           rets[k] = rets[k][..., np.newaxis]  # for concat
       sizes = np.reshape(sizes, [1, 1, 1, -1, 1, 3])
       tile_size_shape = list(rets[0].shape)
       tile_size_shape[3] = 1
       sizes = np.tile(sizes, tile_size_shape)
       rets.insert(3, sizes)
       ret = np.concatenate(rets, axis=-1)
       anchors = np.transpose(ret, [2, 1, 0, 3, 4, 5])
       anchors = anchors.reshape([*anchors.shape[:3], -1, 7])
       anchors_list.append(anchors)
    anchors = np.concatenate(anchors_list, axis=-2)
    anchors = anchors.reshape([-1, 7])
    # print("anchors :", anchors )
    anchors_bv = rbbox2d_to_near_bbox( anchors[:, [0, 1, 3, 4, 6]])
    return anchors, anchors_bv

# 3.3 anchor mask
def sparse_sum_for_anchors_mask(coors, voxel_num, shape):
    ret = np.zeros(shape, dtype=np.float32)
    # print("sparse coors  ", coors.shape, shape) # 12000 3 ,  496 432
    # for i in range(coors.shape[0]):
    for i in range(voxel_num):
        # ret[coors[i, 1], coors[i, 2]] += 1
        ret[coors[i, 2], coors[i, 3]] += 1
    #for i in range(shape[0]):
    #  for j in range(shape[1]):
    #      if ret[i][j] !=0 : print("ret " , i, j, ret[i][j])
    #print("ret sparse :", ret )
    return ret

def fused_get_anchors_area(dense_map, anchors_bv, stride, offset, grid_size):
    # print("fused shape :", anchors_bv.shape,stride, offset,  grid_size) # 321408 4
    anchor_coor = np.zeros(anchors_bv.shape[1:], dtype=np.int32)
    grid_size_x = grid_size[0] - 1
    grid_size_y = grid_size[1] - 1
    N = anchors_bv.shape[0]
    ret = np.zeros((N), dtype=dense_map.dtype)
    for i in range(N):
        anchor_coor[0] = np.floor(
            (anchors_bv[i, 0] - offset[0]) / stride[0])
        anchor_coor[1] = np.floor(
            (anchors_bv[i, 1] - offset[1]) / stride[1])
        anchor_coor[2] = np.floor(
            (anchors_bv[i, 2] - offset[0]) / stride[0])
        anchor_coor[3] = np.floor(
            (anchors_bv[i, 3] - offset[1]) / stride[1])
        anchor_coor[0] = max(anchor_coor[0], 0)
        anchor_coor[1] = max(anchor_coor[1], 0)
        anchor_coor[2] = min(anchor_coor[2], grid_size_x)
        anchor_coor[3] = min(anchor_coor[3], grid_size_y)
        ID = dense_map[anchor_coor[3], anchor_coor[2]]
        IA = dense_map[anchor_coor[1], anchor_coor[0]]
        IB = dense_map[anchor_coor[3], anchor_coor[0]]
        IC = dense_map[anchor_coor[1], anchor_coor[2]]
        ret[i] = ID - IB - IC + IA
        # print("idia : " , ID, IA, IB, IC,  ret[i] )
    #print ("fused ", ret )
    return ret

def get_anchors_mask(coordinates, grid_size, voxel_num):
    coors = coordinates
    dense_voxel_map = sparse_sum_for_anchors_mask(
        coors, voxel_num, tuple(grid_size[::-1][1:]))
    dense_voxel_map = dense_voxel_map.cumsum(0)
    dense_voxel_map = dense_voxel_map.cumsum(1)
    # print("dense " , dense_voxel_map.shape ) 496 432

    #for i in range(dense_voxel_map.shape[0]):
    #  for j in range(dense_voxel_map.shape[1]):
    #      print("ret " , i, j, dense_voxel_map[i][j])
    anchors_area = fused_get_anchors_area(
        dense_voxel_map, g_anchors_bv, voxel_size, point_cloud_range, grid_size)

    # anchors_mask = anchors_area > anchor_area_threshold
    anchors_mask1=[0]* len(anchors_area)
    j=0
    for i in range( len(anchors_area)):
       if anchors_area[i] > anchor_area_threshold:
          anchors_mask1[j]=i
          j=j+1
    anchors_mask = anchors_mask1[:j]
    # print("anchor mask num : " , len(anchors_mask) , anchors_mask )
    return anchors_mask

# 3.4 preprocess
def preprocess_one_data(bin_path, width, height , in_scale):
    with open(bin_path, 'rb') as f:
       data=np.fromfile(f,dtype=np.float32)
    points=np.reshape(data,[-1,4])
    # print("points ", len(points), len(points[0]), points[0,1] ) # 125103 4

    voxels = np.zeros( shape=(max_voxels, max_points, points.shape[-1]), dtype=points.dtype)

    num_points_per_voxel = np.zeros(shape=(max_voxels, ), dtype=np.int32)
    voxelmap_shape = tuple(np.round(grid_size).astype(np.int32).tolist())
    # print("shape ",  np.round(grid_size),  voxelmap_shape) # 432 496 1
    coors = np.zeros(shape=(max_voxels, 4), dtype=np.int32)
    coor_to_voxelidx = -np.ones(shape=voxelmap_shape, dtype=np.int32)
    # print("coor to vo :", coor_to_voxelidx )
    ndim = 3
    voxel_num = 0
    failed = False
    N = points.shape[0]
    # print ("N", N) #  125103
    coor = np.zeros(shape=(4, ), dtype=np.int32)
    for i in range (N):
         failed = False
         for j in range(ndim):
            c = np.floor((points[i,j] - point_cloud_range[j])/voxel_size[j])
            if c<0 or c>=grid_size[j]:
                failed=True
                break
            coor[j] = c
         if failed:
             continue
         # print("coor  " , coor ) # 139 288 0 0
         voxelidx = coor_to_voxelidx[coor[0], coor[1], coor[2]]
         if voxelidx == -1:
             voxelidx = voxel_num
             if voxel_num >= max_voxels:
                 # break
                 continue
             voxel_num += 1
             coor_to_voxelidx[coor[0], coor[1], coor[2]] = voxelidx
             coors[voxelidx] = [coor[3], coor[2], coor[1], coor[0]]
             # print("coors : ", coors[voxelidx] )
         num = num_points_per_voxel[voxelidx]
         if num < max_points:
             voxels[voxelidx, num, 0] = (points[i,0]-point_cloud_range[0])*in_scale/(point_cloud_range[3]-point_cloud_range[0] )
             voxels[voxelidx, num, 1] = (points[i,1]-point_cloud_range[1])*in_scale/(point_cloud_range[4]-point_cloud_range[1] )
             voxels[voxelidx, num, 2] = (points[i,2]-point_cloud_range[2])*in_scale/(point_cloud_range[5]-point_cloud_range[2] )
             voxels[voxelidx, num, 3] = points[i,3]*in_scale
             num_points_per_voxel[voxelidx] += 1
             #print( voxels[voxelidx, num], voxelidx, num, num_points_per_voxel[voxelidx])
    # print("voxel_num ", voxel_num, num_points_per_voxel)
    # print("coors ", coors )
    coor = -np.ones(shape=(4, ), dtype=np.int32)
    coors[voxel_num] = coor
    return voxels.astype(np.int8), coors, voxel_num

# 3.5 post

def sigmoid_array(x):
   return 1 / (1 + np.exp(-x))

def corner_to_standup_nd(boxes_corner):
    ndim = boxes_corner.shape[2]
    assert ndim==2
    # print("ndim :", ndim , boxes_corner[:, : ], np.min( boxes_corner[:, :, 0], axis=1 ))
    standup_boxes = []
    """
    for i in range(ndim):
        standup_boxes.append(torch.min(boxes_corner[:, :, i], dim=1)[0])
    for i in range(ndim):
        standup_boxes.append(torch.max(boxes_corner[:, :, i], dim=1)[0])

    for i in range(boxes_corner.shape[0]):
      standup_boxes.append( np.min(boxes_corner[i] )[0])
      standup_boxes.append( np.max(boxes_corner[i] )[0])
    """
    standup_boxes.append(np.min(boxes_corner[:, :, 0], axis=1 ))
    standup_boxes.append(np.min(boxes_corner[:, :, 1], axis=1 ))
    standup_boxes.append(np.max(boxes_corner[:, :, 0], axis=1 ))
    standup_boxes.append(np.max(boxes_corner[:, :, 1], axis=1 ))
    # print("stand :", standup_boxes )
    ret = np.stack(standup_boxes, axis=1)
    # print("ret :", ret )
    return ret

def nms_jit(dets, scores, thresh, eps=1.0):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    areas = (x2 - x1 + eps) * (y2 - y1 + eps)
    order = scores.argsort()[::-1].astype(np.int32)  # highest->lowest
    ndets = dets.shape[0]
    # suppressed = np.zeros(ndets, dtype=np.int32)
    suppressed = [0]*ndets

    keep = []
    for _i in range(ndets):
        i = order[_i]  # start with highest score box
        if suppressed[i] == 1:
            # if any box have enough iou with this, remove it
            continue
        keep.append(i)
        for _j in range(_i + 1, ndets):
            j = order[_j]
            if suppressed[j] == 1:
                continue
            # calculate iou between i and j box
            w = max(min(x2[i], x2[j]) - max(x1[i], x1[j]) + eps, 0.0)
            h = max(min(y2[i], y2[j]) - max(y1[i], y1[j]) + eps, 0.0)
            inter = w * h
            ovr = inter / (areas[i] + areas[j] - inter)
            # ovr = inter / areas[j]
            if ovr >= thresh:
                suppressed[j] = 1
    # print("keep :", keep )
    return keep


def get_box_decode( idx, box, anchors):
   o = [0]*7
   za = anchors[ idx ][2] + anchors[ idx ][5]/2
   diagonal = math.sqrt( pow(anchors[ idx  ][4], 2.0) + math.pow(anchors[ idx  ][3], 2.0))
   o[0] = box[0] * diagonal + anchors[ idx ][0]
   o[1] = box[1] * diagonal + anchors[ idx ][1]
   o[2] = box[2] * anchors[ idx ][5] + za
   o[3] = math.exp(box[3]) * anchors[ idx ][3]
   o[4] = math.exp(box[4]) * anchors[ idx ][4]
   o[5] = math.exp(box[5]) * anchors[ idx ][5]
   o[6] = box[6] + anchors[ idx ][6]
   o[2] = o[2] - o[5]/2
   # print("o : " , idx, box, o )
   return o;

# 3.6 drawing
#    ignore

# 3.7 main

def get_max(inv):
    if inv[0] > inv[1]:
        return 0 if inv[0] > inv[2] else 2
    else:
        return 1 if inv[1] > inv[2] else 2

def my_test():
    a=math.sqrt(10)
    print ("a", a)
    return

def main(argv):

    #my_test()
    #return
    #np.set_printoptions(threshold=np.inf)

    # init part
    global grid_size
    global g_anchors
    global g_anchors_bv
    grid_size = get_grid_size()
    # print("create_anchor")
    g_anchors, g_anchors_bv = create_anchor(grid_size)
    # print("anchorbv: ", g_anchors_bv)

    # print("create_runner")
    g = xir.Graph.deserialize(argv[1])
    runner = vitis_ai_library.GraphRunner.create_graph_runner(g)
    input_tensor_buffers = runner.get_inputs()
    output_tensor_buffers = runner.get_outputs()

    # for i in range(3): print("name ", output_tensor_buffers[i].get_tensor().name )# dir_cls cls box

    input_ndim = tuple(input_tensor_buffers[1].get_tensor().dims)
    batch = input_ndim[0]
    width = input_ndim[1]
    height = input_ndim[2]
    fixpos = input_tensor_buffers[1].get_tensor().get_attr("fix_point")
    fix_scale = 2**fixpos
    # print("input dim ", input_ndim , fixpos, fix_scale)  #  (1, 12000, 100, 4) 6 64 #  1st is layer0

    # print("preprocess")
    input_ret, coordinates, voxel_num= preprocess_one_data(argv[2], width, height , fix_scale )
    input_data = np.asarray(input_tensor_buffers[1])
    input_data[0] = input_ret
    input_data = np.asarray(input_tensor_buffers[0])
    input_data[0] = coordinates

    # print("get_anchor_mask")
    anchors_mask = get_anchors_mask(coordinates, grid_size, voxel_num)

    """ run graph runner"""
    # print("model run")
    job_id = runner.execute_async(input_tensor_buffers, output_tensor_buffers)
    runner.wait(job_id)

    # print("postprocess ")
    output_size=[1,1,1]
    for i in range(3):
      output_size[i] = int(
        output_tensor_buffers[i].get_tensor().get_element_num() / batch)
    # print("output size :  ",  output_size )
    BOX_IDX = np.argmax(output_size)
    output_size[BOX_IDX] = 0
    CLS_IDX = np.argmax(output_size)
    output_size[CLS_IDX] = 0
    DIR_IDX = np.argmax(output_size)
    output_size[DIR_IDX] = 0
    # print("DIR CLS BOX :", DIR_IDX,CLS_IDX,BOX_IDX)
    output_data = [1,1,1]
    output_data[DIR_IDX] = np.asarray(output_tensor_buffers[DIR_IDX]).reshape(-1,2)
    output_data[CLS_IDX] = np.asarray(output_tensor_buffers[CLS_IDX]).reshape(-1,3)
    output_data[BOX_IDX] = np.asarray(output_tensor_buffers[BOX_IDX]).reshape(-1,7)

    # print("output_d ", output_data[DIR_IDX].shape, output_data[CLS_IDX].shape, output_data[BOX_IDX].shape)
    # (321408, 2) (321408, 3) (321408, 7)
    #  nocreshape ;(1, 248, 216, 12) (1, 248, 216, 18) (1, 248, 216, 42)
    box_preds1 = output_data[BOX_IDX]
    cls_preds1 = output_data[CLS_IDX]
    dir_preds1 = output_data[DIR_IDX]
    # print("dir cls box _preds ", dir_preds.shape , cls_preds.shape, box_preds.shape) # [42378 2]

    predictions_dicts = []

    nms_confidence1 = -math.log(1.0/nms_confidence -1.0 )
    # print("nms con:", nms_confidence1)
    box_preds  = []
    cls_preds  = []
    dir_labels = []
    top_labels = []
    # print(" len anchorsmask: " , len(anchors_mask ))
    for i in range(len(anchors_mask)):
       ii = anchors_mask[i]
       # print("ii " , ii )
       pos = get_max(cls_preds1[ii])
       if cls_preds1[ii][pos] >= nms_confidence1:
          box_preds.append(get_box_decode( ii,box_preds1[ii] , g_anchors ))
          cls_preds.append(cls_preds1[ii][pos])
          dir_labels.append( 0 if dir_preds1[ii][0] >= dir_preds1[ii][1] else 1 )
          top_labels.append(pos)

    # print("top_labels :", len(top_labels ))
    box_preds=np.array(box_preds)
    boxes_for_nms = box_preds[:, [0, 1, 3, 4, 6]]
    box_preds_corners = center_to_corner_box2d(
      boxes_for_nms[:, :2],
      boxes_for_nms[:, 2:4],
      boxes_for_nms[:, 4])
    # print("box_preds_corners :", box_preds_corners.shape) # 30 4 2
    boxes_for_nms = corner_to_standup_nd( box_preds_corners)

    # next, nms
    cls_preds=np.array(cls_preds)
    selected = nms_jit(boxes_for_nms, cls_preds,nms_iou_threshold )

    # detected result
    label_preds  = []
    scores_preds = []
    boxes_preds  = []
    for i in range(len(selected)):
        label_preds.append(top_labels[selected[i]])
        scores_preds.append( 1.0/(1.0+math.exp(-1.0* cls_preds[selected[i]])))
        # print(" xor :",  bool(box_preds[selected[i] ][ 6 ] >0) ,  bool(dir_labels[selected[i]]))
        if  bool(box_preds[selected[i] ][ 6 ] >0) ^ bool(dir_labels[selected[i]])  :
          box_preds[ selected[i]][ 6 ] += 3.14159265 ;
        boxes_preds.append(box_preds[ selected[i]])

    # show result
    # print("label_preds  boxes  scores  : ")
    for i in range(len(selected)):
        print( label_preds[i], boxes_preds[i],  scores_preds[i] )

    del runner


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("please model file and input file.")
    else:
        main(sys.argv)

