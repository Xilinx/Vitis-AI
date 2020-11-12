
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

from enum import Enum

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pyqtgraph.opengl as gl
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
from pyqtgraph.opengl.GLGraphicsItem import GLGraphicsItem
from pyqtgraph.Qt import QtCore, QtGui

class FORMAT(Enum):
    """enum that indicate format of a bbox
    """
    Center = 'format_bbox_centor'
    Corner = 'format_bbox_corner'
    Length = 'format_bbox_length'


class GLColor(Enum):
    Red = (1.0, 0.0, 0.0)
    Lime = (0.0, 1.0, 0.0)
    Green = (0.0, 0.5, 0.0)
    Blue = (0.0, 0.0, 1.0)
    Gray = (0.5, 0.5, 0.5)
    Yellow = (1.0, 1.0, 0.0)
    Write = (1.0, 1.0, 1.0)
    Cyan = (0.0, 1.0, 1.0)
    Magenta = (1.0, 0.0, 1.0)
    Silver = (0.75, 0.75, 0.75)
    Maroon = (0.5, 0.0, 0.0)
    Olive = (0.5, 0.5, 0.0)
    Teal = (0.0, 0.5, 0.5)
    Navy = (0.0, 0.0, 0.5)
    Purple = (0.5, 0.0, 0.5)


def gl_color(color, alpha=1.0):
    return (*color.value, alpha)


def corner_to_length(bboxes):
    if len(bboxes.shape) == 1:
        bboxes = bboxes.reshape([1, -1])
    ndim = bboxes.shape[1] // 2
    return np.concatenate(
        [bboxes[:, :ndim], bboxes[:, ndim:] - bboxes[:, :ndim]], axis=1)


def draw_bbox_in_ax(ax,
                    bboxes,
                    rotations=None,
                    fmt=FORMAT.Corner,
                    labels=None,
                    label_size='small',
                    edgecolors='r',
                    linestyle='dashed',
                    alpha=0.5):
    if rotations is None:
        rotations = np.zeros([bboxes.shape[0]])
    else:
        rotations = rotations / np.pi * 180
    if labels is None:
        labels = [None] * bboxes.shape[0]
    if not isinstance(edgecolors, list):
        edgecolors = [edgecolors for i in range(len(bboxes))]
    if fmt == FORMAT.Corner:
        bboxes = corner_to_length(bboxes)
    for bbox, rot, e_color, label in zip(bboxes, rotations, edgecolors,
                                         labels):
        rect_p = patches.Rectangle(
            bbox[:2],
            bbox[2],
            bbox[3],
            rot,
            fill=False,
            edgecolor=e_color,
            linestyle=linestyle,
            alpha=alpha)
        ax.add_patch(rect_p)
        if label is not None:
            t = ax.text(
                bbox[0],
                bbox[1],
                label,
                ha="left",
                va="bottom",
                color=e_color,
                size=label_size)
    return ax



def draw_3d_bbox_in_ax(ax, bboxes, colors='r', alpha=0.5, image_shape=None):
    # assume bboxes has right format.
    if not isinstance(colors, list):
        colors = [colors for i in range(len(bboxes))]
    for box, color in zip(bboxes, colors):
        box_a, box_b = box[:4], box[4:]
        for pa, pb in zip(box_a, box_a[[1, 2, 3, 0]]):
            ax.plot([pa[0], pb[0]], [pa[1], pb[1]], color=color, alpha=alpha)
        for pa, pb in zip(box_b, box_b[[1, 2, 3, 0]]):
            ax.plot([pa[0], pb[0]], [pa[1], pb[1]], color=color, alpha=alpha)
        for pa, pb in zip(box_a, box_b):
            ax.plot([pa[0], pb[0]], [pa[1], pb[1]], color=color, alpha=alpha)
    if image_shape is not None:
        patch = patches.Rectangle([0, 0], image_shape[1], image_shape[0])
        ax.set_clip_path(patch)
    return ax


def draw_2d_bbox_in_ax(ax,
                       bboxes,
                       colors='r',
                       alpha=0.5,
                       with_arrow=True,
                       behind_axes=[0, 1]):
    # assume bboxes has right format. [N, 4, 2]
    if not isinstance(colors, list):
        colors = [colors for i in range(len(bboxes))]
    for box, color in zip(bboxes, colors):
        for pa, pb in zip(box, box[[1, 2, 3, 0]]):
            ax.plot([pa[0], pb[0]], [pa[1], pb[1]], color=color, alpha=alpha)
        if with_arrow:
            center = np.mean(box, axis=0)
            start = np.mean(
                np.concatenate([center[np.newaxis, ...], box[behind_axes]]),
                axis=0)
            front_axes = [i for i in range(4) if i not in behind_axes]
            end = np.mean(
                np.concatenate([center[np.newaxis, ...], box[front_axes]]),
                axis=0)
            ax.arrow(
                start[0],
                start[1],
                end[0] - start[0],
                end[1] - start[1],
                head_width=0.2,
                head_length=0.2,
                fc=color,
                ec=color)
    return ax


def draw_3d_bbox_in_3dax(ax, bboxes, colors='r', alpha=0.25, facecolors=None):
    if not isinstance(colors, list):
        colors = [colors for i in range(len(bboxes))]
    if not isinstance(facecolors, list):
        facecolors = [facecolors for i in range(len(bboxes))]

    for box, color, facecolor in zip(bboxes, colors, facecolors):
        ax.scatter3D(box[:, 0], box[:, 1], box[:, 2], marker='.', color=color)
        verts = np.array([
            [box[0], box[1], box[2], box[3]],
            [box[4], box[5], box[6], box[7]],
            [box[0], box[3], box[7], box[4]],
            [box[1], box[2], box[6], box[5]],
            [box[0], box[1], box[5], box[4]],
            [box[3], box[2], box[6], box[7]],
        ])
        mp3dcoll = Poly3DCollection(
            verts,
            linewidths=1,
            edgecolors=color,
            alpha=alpha,
            facecolors=facecolor)
        mp3dcoll.set_facecolor(facecolor)
        mp3dcoll.set_edgecolor(color)
        mp3dcoll.set_alpha(alpha)
        ax.add_collection3d(mp3dcoll)
    return ax


class GLTextItem(GLGraphicsItem):
    def __init__(self, X=None, Y=None, Z=None, text=None, color=None):
        GLGraphicsItem.__init__(self)
        self.color = color
        if color is None:
            self.color = QtCore.Qt.white
        self.text = text
        self.X = X
        self.Y = Y
        self.Z = Z

    def setGLViewWidget(self, GLViewWidget):
        self.GLViewWidget = GLViewWidget

    def setText(self, text):
        self.text = text
        self.update()

    def setX(self, X):
        self.X = X
        self.update()

    def setY(self, Y):
        self.Y = Y
        self.update()

    def setZ(self, Z):
        self.Z = Z
        self.update()

    def paint(self):
        self.GLViewWidget.qglColor(self.color)
        self.GLViewWidget.renderText(self.X, self.Y, self.Z, self.text)


class GLLabelItem(GLGraphicsItem):
    def __init__(self, pos=None, text=None, color=None, font=QtGui.QFont()):
        GLGraphicsItem.__init__(self)
        self.color = color
        if color is None:
            self.color = QtCore.Qt.white
        self.text = text
        self.pos = pos
        self.font = font
        self.font.setPointSizeF(20)

    def setGLViewWidget(self, GLViewWidget):
        self.GLViewWidget = GLViewWidget

    def setData(self, pos, text, color):
        self.text = text
        self.pos = pos
        self.color = color
        self.update()

    def paint(self):
        self.GLViewWidget.qglColor(self.color)
        if self.pos is not None and self.text is not None:
            if isinstance(self.pos, (list, tuple, np.ndarray)):
                for p, text in zip(self.pos, self.text):
                    self.GLViewWidget.renderText(*p, text, self.font)
            else:
                self.GLViewWidget.renderText(*self.pos, self.text, self.font)


def _pltcolor_to_qtcolor(color):
    color_map = {
        'r': QtCore.Qt.red,
        'g': QtCore.Qt.green,
        'b': QtCore.Qt.blue,
        'k': QtCore.Qt.black,
        'w': QtCore.Qt.white,
        'y': QtCore.Qt.yellow,
        'c': QtCore.Qt.cyan,
        'm': QtCore.Qt.magenta,
    }
    return color_map[color]


from second.core.box_np_ops import minmax_to_corner_3d


def draw_bounding_box(widget, box_minmax, color):
    bbox = minmax_to_corner_3d(box_minmax)
    return draw_3d_bboxlines_in_pyqt(widget, bbox, color)


def draw_3d_bboxlines_in_pyqt(widget,
                              bboxes,
                              colors=GLColor.Green,
                              width=1.0,
                              labels=None,
                              alpha=1.0,
                              label_color='r',
                              line_item=None,
                              label_item=None):
    if bboxes.shape[0] == 0:
        bboxes = np.zeros([0, 8, 3])
    if not isinstance(colors, (list, np.ndarray)):
        if isinstance(colors, GLColor):
            colors = gl_color(colors, alpha)
        colors = [colors for i in range(len(bboxes))]
    if not isinstance(labels, (list, np.ndarray)):
        labels = [labels for i in range(len(bboxes))]
    total_lines = []
    total_colors = []
    for box, facecolor in zip(bboxes, colors):
        lines = np.array([
            box[0], box[1], box[1], box[2], box[2], box[3], box[3], box[0],
            box[1], box[5], box[5], box[4], box[4], box[0], box[2], box[6],
            box[6], box[7], box[7], box[3], box[5], box[6], box[4], box[7]
        ])
        total_lines.append(lines)
        color = np.array([list(facecolor) for i in range(len(lines))])
        total_colors.append(color)
    if bboxes.shape[0] != 0:
        total_lines = np.concatenate(total_lines, axis=0)
        total_colors = np.concatenate(total_colors, axis=0)
    else:
        total_lines = None
        total_colors = None
    if line_item is None:
        line_item = gl.GLLinePlotItem(
            pos=total_lines,
            color=total_colors,
            width=width,
            antialias=True,
            mode='lines')
        widget.addItem(line_item)
    else:
        line_item.setData(
            pos=total_lines,
            color=total_colors,
            width=width,
            antialias=True,
            mode='lines')
    label_color_qt = _pltcolor_to_qtcolor(label_color)
    if labels is not None:
        if label_item is None:
            label_item = GLLabelItem(bboxes[:, 0, :], labels, label_color_qt)
            label_item.setGLViewWidget(widget)
            widget.addItem(label_item)
        else:
            label_item.setData(
                pos=bboxes[:, 0, :], text=labels, color=label_color_qt)
    """
    for box, label in zip(bboxes, labels):
        if label is not None:
            label_color_qt = _pltcolor_to_qtcolor(label_color)
            t = GLTextItem(
                X=box[0, 0],
                Y=box[0, 1],
                Z=box[0, 2],
                text=label,
                color=label_color_qt)
            t.setGLViewWidget(widget)
            widget.addItem(t)
    """
    return line_item, label_item


def draw_bboxlines_in_pyqt(widget,
                           bboxes,
                           colors=GLColor.Green,
                           width=1.0,
                           labels=None,
                           alpha=1.0,
                           label_color='r',
                           line_item=None,
                           label_item=None):
    if bboxes.shape[0] == 0:
        return
    if not isinstance(colors, list):
        if isinstance(colors, GLColor):
            colors = gl_color(colors, alpha)
        colors = [colors for i in range(len(bboxes))]
    if not isinstance(labels, list):
        labels = [labels for i in range(len(bboxes))]
    total_lines = []
    total_colors = []
    for box, facecolor in zip(bboxes, colors):
        lines = np.array(
            [box[0], box[1], box[1], box[2], box[2], box[3], box[3], box[0]])
        total_lines.append(lines)
        color = np.array([list(facecolor) for i in range(len(lines))])
        total_colors.append(color)
    total_lines = np.concatenate(total_lines, axis=0)
    total_colors = np.concatenate(total_colors, axis=0)
    if line_item is None:
        line_item = gl.GLLinePlotItem(
            pos=total_lines,
            color=total_colors,
            width=width,
            antialias=True,
            mode='lines')
        widget.addItem(line_item)
    else:
        line_item.setData(
            pos=total_lines,
            color=total_colors,
            width=width,
            antialias=True,
            mode='lines')
    label_color_qt = _pltcolor_to_qtcolor(label_color)
    if labels is not None:
        if label_item is None:
            label_item = GLLabelItem(bboxes[:, 0, :], labels, label_color_qt)
            label_item.setGLViewWidget(widget)
            widget.addItem(label_item)
        else:
            label_item.setData(
                pos=bboxes[:, 0, :], text=labels, color=label_color_qt)

    return line_item, label_item


def _3d_bbox_to_mesh(bboxes):
    bbox_faces = np.array([
        [0, 1, 2],
        [0, 2, 3],
        [4, 5, 6],
        [4, 6, 7],
        [0, 4, 7],
        [0, 7, 3],
        [1, 5, 6],
        [1, 6, 2],
        [3, 2, 6],
        [3, 6, 7],
        [0, 1, 5],
        [0, 5, 4],
    ])
    verts_list = []
    faces_list = []
    for i, bbox in enumerate(bboxes):
        # bbox: [8, 3]
        verts_list.append(bbox)
        faces_list.append(bbox_faces + 8 * i)
    verts = np.concatenate(verts_list, axis=0)
    faces = np.concatenate(faces_list, axis=0)
    return verts, faces


def draw_3d_bbox_meshes_in_pyqt(widget,
                                bboxes,
                                colors=GLColor.Gray,
                                alpha=1.0,
                                edgecolors=None):
    verts, faces = _3d_bbox_to_mesh(bboxes)
    if not isinstance(colors, list):
        if isinstance(colors, GLColor):
            colors = gl_color(colors, alpha)
        colors = np.array([colors for i in range(len(verts))])
    m1 = gl.GLMeshItem(
        vertexes=verts, faces=faces, faceColors=colors, smooth=False)
    m1.setGLOptions('additive')
    widget.addItem(m1)
    return widget


def draw_3d_bboxlines_in_pyqt_v1(widget,
                                 bboxes,
                                 colors=(0.0, 1.0, 0.0, 1.0),
                                 width=1.0,
                                 labels=None,
                                 label_color='r'):
    if not isinstance(colors, list):
        colors = [colors for i in range(len(bboxes))]
    if not isinstance(labels, list):
        labels = [labels for i in range(len(bboxes))]
    for box, facecolor, label in zip(bboxes, colors, labels):
        lines = np.array([
            box[0], box[1], box[1], box[2], box[2], box[3], box[3], box[0],
            box[1], box[5], box[5], box[4], box[4], box[0], box[2], box[6],
            box[6], box[7], box[7], box[3], box[5], box[6], box[4], box[7]
        ])
        color = np.array([list(facecolor) for i in range(len(lines))])
        plt = gl.GLLinePlotItem(
            pos=lines, color=color, width=width, antialias=True, mode='lines')
        widget.addItem(plt)
        if label is not None:
            label_color_qt = _pltcolor_to_qtcolor(label_color)
            t = GLTextItem(
                X=box[0, 0],
                Y=box[0, 1],
                Z=box[0, 2],
                text=label,
                color=label_color_qt)
            t.setGLViewWidget(widget)
            widget.addItem(t)

    return widget
