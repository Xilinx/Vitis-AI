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

"""
The MIT License

Copyright (c) 2019 Andres Milioto, Jens Behley, Cyrill Stachniss, Photogrammetry and Robotics Lab, University of Bonn.
Copyright (c) 2020 Tiago Cortinhal, George Tzelepis and Eren Erdal Aksoy

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

import os

import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from concurrent import futures
import yaml
import pdb

import pykitti

basedir = '../../../Semantic-KITTI/dataset/'
sequence = '01'
uncerts = '/predictions/uncert/'
preds = '/predictions/preds/'
gt = '/labels'
img = '/image_2'
lidar = '/velodyne'
projected_uncert = '/projected_images/uncert/'
projected_preds = '/projected_images/preds/'

dataset = pykitti.odometry(basedir, sequence)

EXTENSIONS_LABEL = ['.label']
EXTENSIONS_LIDAR = ['.bin']
EXTENSIONS_IMG = ['.png']


def is_label(filename):
    return any(filename.endswith(ext) for ext in EXTENSIONS_LABEL)


def is_lidar(filename):
    return any(filename.endswith(ext) for ext in EXTENSIONS_LIDAR)


def is_img(filename):
    return any(filename.endswith(ext) for ext in EXTENSIONS_IMG)


def get_mpl_colormap(cmap_name):
    cmap = plt.get_cmap(cmap_name)

    # Initialize the matplotlib color map
    sm = plt.cm.ScalarMappable(cmap=cmap)

    # Obtain linear color range
    color_range = sm.to_rgba(np.linspace(0, 1, 256), bytes=True)[:, 2::-1]

    return color_range.reshape(256, 3).astype(np.float32) / 255.0


path = os.path.join(basedir + 'sequences/' + sequence + uncerts)

scan_uncert = [os.path.join(dp, f) for dp, dn, fn in os.walk(
    os.path.expanduser(path)) for f in fn if is_label(f)]
scan_uncert.sort()

path = os.path.join(basedir + 'sequences/' + sequence + preds)
scan_preds = [os.path.join(dp, f) for dp, dn, fn in os.walk(
    os.path.expanduser(path)) for f in fn if is_label(f)]
scan_preds.sort()

path = os.path.join(basedir + 'sequences/' + sequence + gt)
scan_gt = [os.path.join(dp, f) for dp, dn, fn in os.walk(
    os.path.expanduser(path)) for f in fn if is_label(f)]
scan_gt.sort()

color_map_dict = yaml.safe_load(open("color_map.yml"))['color_map']
color_map = {}
for key, value in color_map_dict.items():
    color_map[key] = np.array(value, np.float32) / 255.0


def plot_and_save(label_uncert, label_name, lidar_name, cam2_image_name):
    labels = np.fromfile(label_name, dtype=np.int32).reshape((-1))
    uncerts = np.fromfile(label_uncert, dtype=np.float32).reshape((-1))
    uncerts = ((uncerts - min(uncerts)) / (max(uncerts) - min(uncerts)))
    velo_points = np.fromfile(lidar_name, dtype=np.float32).reshape(-1, 4)
    try:
        cam2_image = plt.imread(cam2_image_name)
    except IOError:
        print('detect error img %s' % label_name)

    plt.imshow(cam2_image)

    if True:

        # Project points to camera.
        cam2_points = dataset.calib.T_cam2_velo.dot(velo_points.T).T

        # Filter out points behind camera
        idx = cam2_points[:, 2] > 0
        # velo_points_projected = velo_points[idx]
        cam2_points = cam2_points[idx]
        labels_projected = labels[idx]
        uncert_projected = uncerts[idx]

        # Remove homogeneous z.
        cam2_points = cam2_points[:, :3] / cam2_points[:, 2:3]

        # Apply instrinsics.
        intrinsic_cam2 = dataset.calib.K_cam2
        cam2_points = intrinsic_cam2.dot(cam2_points.T).T[:, [1, 0]]
        cam2_points = cam2_points.astype(int)

        for i in range(0, cam2_points.shape[0]):
            u, v = cam2_points[i, :]
            label = labels_projected[i]
            uncert = uncert_projected[i]
            if label > 0 and v > 0 and v < cam2_image.shape[1] and u > 0 and u < cam2_image.shape[0]:
                m_circle = plt.Circle((v, u), 1,
                                      # color=matplotlib.cm.viridis(uncert),
                                      alpha=0.4,
                                      color=color_map[label][..., ::-1]
                                      )
                plt.gcf().gca().add_artist(m_circle)

    plt.axis('off')
    path = os.path.join(basedir + 'sequences/' + sequence + projected_preds)
    plt.savefig(path + label_name.split('/')[-1].split('.')[0] + '.png', bbox_inches='tight', transparent=True,
                pad_inches=0)
    # plt.show()
    print(label_name.split('/')[-1])
    plt.clf()


with futures.ProcessPoolExecutor() as pool:
    for label_uncert, label_name, lidar_name, cam2_image_name in zip(scan_uncert, scan_preds, dataset.velo_files,
                                                                     dataset.cam2_files):
        print(label_name)
        # if label_name == '/SPACE/DATA/SemanticKITTI/dataset/sequences/13/predictions/preds/001032.label':
        plot_and_save(label_uncert, label_name, lidar_name, cam2_image_name)

if __name__ == "__main__":
    pass
