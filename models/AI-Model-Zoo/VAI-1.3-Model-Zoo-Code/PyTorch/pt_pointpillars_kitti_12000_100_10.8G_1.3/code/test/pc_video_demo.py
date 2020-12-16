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

import os
import cv2
import numpy as np
import mayavi.mlab as mlab
import datasets.kitti_util as utils
from datasets.kitti_object import kitti_object_video

try:
    raw_input  # Python 2
except NameError:
    raw_input = input  # Python 3


fig = mlab.figure(
    figure=None, bgcolor=(0, 0, 0), fgcolor=None, engine=None, size=(1600, 800)
)


def draw_lidar(
    pc,
    color=None,
    fig=None,
    bgcolor=(0, 0, 0),
    pts_scale=0.3,
    pts_mode="sphere",
    pts_color=None,
    color_by_intensity=False,
    pc_label=False,
    pc_range=[],
):
    """ Draw lidar points
    Args:
        pc: numpy array (n,3) of XYZ
        color: numpy array (n) of intensity or whatever
        fig: mayavi figure handler, if None create new one otherwise will use it
    Returns:
        fig: created or used fig
    """
    
    xmin, xmax, ymin, ymax, zmin, zmax = pc_range
    pts_mode = "point"
    print("====================", pc.shape)
    if fig is None:
        fig = mlab.figure(
            figure=None, bgcolor=bgcolor, fgcolor=None, engine=None, size=(1600, 1000)
        )
    
    if color is None:
        color = pc[:, 2]

    mlab.points3d(
        pc[:, 0],
        pc[:, 1],
        pc[:, 2],
        color,
        color=pts_color,
        mode=pts_mode,
        colormap="gnuplot",
        scale_factor=pts_scale,
        figure=fig,
        vmax=zmax,
        vmin=zmin,
    )

    # draw origin
    mlab.points3d(0, 0, 0, color=(1, 1, 1), mode="sphere", scale_factor=0.2)

    # draw axis
    axes = np.array(
        [[2.0, 0.0, 0.0, 0.0], [0.0, 2.0, 0.0, 0.0], [0.0, 0.0, 2.0, 0.0]],
        dtype=np.float64,
    )
    mlab.plot3d(
        [0, axes[0, 0]],
        [0, axes[0, 1]],
        [0, axes[0, 2]],
        color=(1, 0, 0),
        tube_radius=None,
        figure=fig,
    )
    mlab.plot3d(
        [0, axes[1, 0]],
        [0, axes[1, 1]],
        [0, axes[1, 2]],
        color=(0, 1, 0),
        tube_radius=None,
        figure=fig,
    )
    mlab.plot3d(
        [0, axes[2, 0]],
        [0, axes[2, 1]],
        [0, axes[2, 2]],
        color=(0, 0, 1),
        tube_radius=None,
        figure=fig,
    )

    # draw fov (todo: update to real sensor spec.)
    a_ymin = abs(ymin)
    a_ymax = abs(ymax)
    fov = np.array(
        [[a_ymax, a_ymax, 0.0, 0.0], [a_ymin, -a_ymin, 0.0, 0.0]], dtype=np.float64  # 45 degree
    )

    mlab.plot3d(
        [0, fov[0, 0]],
        [0, fov[0, 1]],
        [0, fov[0, 2]],
        color=(1, 1, 1),
        tube_radius=None,
        line_width=1,
        figure=fig,
    )
    mlab.plot3d(
        [0, fov[1, 0]],
        [0, fov[1, 1]],
        [0, fov[1, 2]],
        color=(1, 1, 1),
        tube_radius=None,
        line_width=1,
        figure=fig,
    )

    # draw square region
    TOP_Y_MIN = ymin
    TOP_Y_MAX = ymax
    TOP_X_MIN = xmin
    TOP_X_MAX = xmax
    #TOP_Z_MIN = -2.0
    #TOP_Z_MAX = 0.4

    x1 = TOP_X_MIN
    x2 = TOP_X_MAX
    y1 = TOP_Y_MIN
    y2 = TOP_Y_MAX
    mlab.plot3d(
        [x1, x1],
        [y1, y2],
        [0, 0],
        color=(0.5, 0.5, 0.5),
        tube_radius=0.1,
        line_width=1,
        figure=fig,
    )
    mlab.plot3d(
        [x2, x2],
        [y1, y2],
        [0, 0],
        color=(0.5, 0.5, 0.5),
        tube_radius=0.1,
        line_width=1,
        figure=fig,
    )
    mlab.plot3d(
        [x1, x2],
        [y1, y1],
        [0, 0],
        color=(0.5, 0.5, 0.5),
        tube_radius=0.1,
        line_width=1,
        figure=fig,
    )
    mlab.plot3d(
        [x1, x2],
        [y2, y2],
        [0, 0],
        color=(0.5, 0.5, 0.5),
        tube_radius=0.1,
        line_width=1,
        figure=fig,
    )

    # mlab.orientation_axes()
    mlab.view(
        azimuth=180,
        elevation=70,
        focalpoint=[12.0909996, -1.04700089, -2.03249991],
        distance=62.0,
        figure=fig,
    )
    return fig


def draw_gt_boxes3d(
    gt_boxes3d,
    fig,
    color=(1, 1, 1),
    line_width=1,
    draw_text=True,
    text_scale=(1, 1, 1),
    color_list=None,
    label=""
):
    """ Draw 3D bounding boxes
    Args:
        gt_boxes3d: numpy array (n,8,3) for XYZs of the box corners
        fig: mayavi figure handler
        color: RGB value tuple in range (0,1), box line color
        line_width: box line width
        draw_text: boolean, if true, write box indices beside boxes
        text_scale: three number tuple
        color_list: a list of RGB tuple, if not None, overwrite color.
    Returns:
        fig: updated fig
    """
    num = len(gt_boxes3d)
    for n in range(num):
        b = gt_boxes3d[n]
        if color_list is not None:
            color = color_list[n]
        if draw_text:
            mlab.text3d(
                b[4, 0],
                b[4, 1],
                b[4, 2],
                label,
                scale=text_scale,
                color=color,
                figure=fig,
            )
        for k in range(0, 4):
            # http://docs.enthought.com/mayavi/mayavi/auto/mlab_helper_functions.html
            i, j = k, (k + 1) % 4
            mlab.plot3d(
                [b[i, 0], b[j, 0]],
                [b[i, 1], b[j, 1]],
                [b[i, 2], b[j, 2]],
                color=color,
                tube_radius=None,
                line_width=line_width,
                reset_zoom=False,
                figure=fig,
            )

            i, j = k + 4, (k + 1) % 4 + 4
            mlab.plot3d(
                [b[i, 0], b[j, 0]],
                [b[i, 1], b[j, 1]],
                [b[i, 2], b[j, 2]],
                color=color,
                tube_radius=None,
                line_width=line_width,
                reset_zoom=False,
                figure=fig,
            )

            i, j = k, k + 4
            mlab.plot3d(
                [b[i, 0], b[j, 0]],
                [b[i, 1], b[j, 1]],
                [b[i, 2], b[j, 2]],
                color=color,
                tube_radius=None,
                line_width=line_width,
                reset_zoom=False,
                figure=fig,
            )
    # mlab.show(1)
    # mlab.view(azimuth=180, elevation=70, focalpoint=[ 12.0909996 , -1.04700089, -2.03249991], distance=62.0, figure=fig)
    return fig


def get_point_index_in_range(pc_velo, xyz_range):
    keep_index = (
        (pc_velo[:, 0] >= xyz_range[0])
        & (pc_velo[:, 0] <= xyz_range[1])
        & (pc_velo[:, 1] >= xyz_range[2])
        & (pc_velo[:, 1] <= xyz_range[3])
        & (pc_velo[:, 2] >= xyz_range[4])
        & (pc_velo[:, 2] <= xyz_range[5])
    )
    return keep_index


def get_lidar_index_in_image_fov(
    pc_velo, calib, xmin, ymin, xmax, ymax, return_more=False, clip_distance=2.0
):
    """ Filter lidar points, keep those in image FOV """
    pts_2d = calib.project_velo_to_image(pc_velo)
    fov_inds = (
        (pts_2d[:, 0] < xmax)
        & (pts_2d[:, 0] >= xmin)
        & (pts_2d[:, 1] < ymax)
        & (pts_2d[:, 1] >= ymin)
    )
    fov_inds = fov_inds & (pc_velo[:, 0] > clip_distance)
    return fov_inds


def get_point_index_with_filters(pc_velo, xyz_range=[], 
                                 img_fov=False, calib=None, img_width=None, img_height=None):
    keep_index = (pc_velo[:, 0] == pc_velo[:, 0])
    if xyz_range:
        pc_velo_index = get_point_index_in_range(pc_velo, xyz_range)
        keep_index &= pc_velo_index
        # pc_velo = pc_velo[pc_velo_index, :]
    if img_fov:
        pc_velo_index = get_lidar_index_in_image_fov(pc_velo[:, :3], calib, 0, 0, img_width, img_height)
        keep_index &= pc_velo_index
        # pc_velo = pc_velo[pc_velo_index, :]
    # print("num of points after filtered: {}", pc_velo.shape[0])
    return keep_index

def draw_pred_boxes3d(objects_pred, calib, colormap, fig=None, score_thresh=0.5):
    if objects_pred is None:
        return
    for obj in objects_pred:
        if obj.type == "DontCare":
            continue
        if obj.score < 0.5:
            continue
        color = colormap[obj.type]
        # Draw 3d bounding box
        _, box3d_pts_3d = utils.compute_box_3d(obj, calib.P)
        box3d_pts_3d_velo = calib.project_rect_to_velo(box3d_pts_3d)
        print("box3d_pts_3d_velo:")
        print(box3d_pts_3d_velo)
        draw_gt_boxes3d([box3d_pts_3d_velo], fig=fig, color=color, draw_text=False)
        # Draw heading arrow
        _, ori3d_pts_3d = utils.compute_orientation_3d(obj, calib.P)
        ori3d_pts_3d_velo = calib.project_rect_to_velo(ori3d_pts_3d)
        x1, y1, z1 = ori3d_pts_3d_velo[0, :]
        x2, y2, z2 = ori3d_pts_3d_velo[1, :]
        mlab.plot3d(
            [x1, x2],
            [y1, y2],
            [z1, z2],
            color=color,
            tube_radius=None,
            line_width=1,
            reset_zoom=False,
            figure=fig,
        )

def viz_kitti_video():
    section = '2011_09_26_drive_0095_sync'
    video_path = section + "/2011_09_26"
    # res_video_path = section + '-multiclass-quant.avi'
    res_video_path = section + '-car-quant.avi'
    fps = 10
    # vxyz_range = [0,70.4,-40,40,-3,2]
    xyz_range = [-70.4,70.4,-40,40,-3,2]
    img_fov = False
    colormap = {'Car': (0,1,0), 'Pedestrian': (1,0,0), 'Cyclist': (0,0,1)}
    score_thresh=0.5

    dataset = kitti_object_video(
        video_path + "/" + section + "/image_02/data",
        video_path + "/" + section + "/velodyne_points/data",
        video_path,
        video_path + "/" + section + "/car_preds",
    )

    f = mlab.gcf()
    f.scene._lift()
    h, w, c = mlab.screenshot().shape
    
    video_writer = cv2.VideoWriter(res_video_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (w, h))

    # mlab.axes().axes.fly_mode = 'none'

    print(len(dataset))
    calib = dataset.get_calibration(None)
    for i in range(len(dataset)):
        img = dataset.get_image(i)
        pc = dataset.get_lidar(i)
        img_height, img_width, _ = img.shape
        keep_index = get_point_index_with_filters(pc, xyz_range, img_fov, calib, img_width, img_height)
        pc = pc[keep_index]
        draw_lidar(pc, fig=fig, pc_range=xyz_range)

        objects_pred = dataset.get_pred_objects(i)
        draw_pred_boxes3d(objects_pred, calib, colormap, fig, score_thresh)

        '''
        f = mlab.gcf()
        camera = f.scene.camera
        camera.zoom(2)
        '''
        # mlab.savefig('tmp_dir/pc_3d_view-{}.png'.format(str(i).zfill(6)), figure=fig)
        f = mlab.gcf()
        f.scene._lift()
        pc_img = mlab.screenshot(antialiased=True)
        print('shape: {}'.format(pc_img.shape))
        video_writer.write(pc_img[:, :, ::-1])
        # mlab.show()
        # input_str = raw_input()
        mlab.clf()
    return

viz_kitti_video()
