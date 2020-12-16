
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

import numpy as np
import pyqtgraph.opengl as gl
from PyQt5 import QtCore, QtGui
from PyQt5.QtCore import QTimer, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QIcon, QMouseEvent, QPainter
from PyQt5.QtWidgets import (QApplication, QCheckBox, QComboBox, QDialog,
                             QFormLayout, QGroupBox, QHBoxLayout, QLabel,
                             QLineEdit, QMainWindow, QPlainTextEdit,
                             QPushButton, QSizePolicy, QVBoxLayout, QWidget)
from pyqtgraph.opengl.GLGraphicsItem import GLGraphicsItem

import OpenGL.GL as pygl

from second.utils.bbox_plot import GLColor

class GLLabelItem(GLGraphicsItem):
    def __init__(self, pos=None, text=None, color=None, font=QtGui.QFont()):
        GLGraphicsItem.__init__(self)
        self.color = color
        if color is None:
            self.color = QtCore.Qt.white
        self.text = text
        self.pos = pos
        self.font = font
        

    def setGLViewWidget(self, GLViewWidget):
        self.GLViewWidget = GLViewWidget
    
    def setData(self, pos, text, color, font):
        self.text = text
        self.pos = pos
        self.color = color
        self.font = font
        self.update()

    def paint(self):
        self.GLViewWidget.qglColor(self.color)
        if self.pos is not None and self.text is not None:
            if isinstance(self.pos, (list, tuple, np.ndarray)):
                for p, text in zip(self.pos, self.text):
                    self.GLViewWidget.renderText(*p, text, self.font)
            else:
                self.GLViewWidget.renderText(*self.pos, self.text, self.font)


def gl_color(color, alpha=1.0):
    return (*color.value, alpha)


def _extend_color_if_necessary(colors, shape, alphas):
    if not isinstance(shape, (list, tuple)):
        shape = [shape]
    num = np.prod(shape)
    if isinstance(colors, (list, tuple, np.ndarray)):
        colors_np = np.array(colors)
        if colors_np.shape[0] == 4 and len(colors_np.shape) == 1:
            colors = (*colors,)
        elif colors_np.shape[0] == 3 and len(colors_np.shape) == 1:
            colors = (*colors, 1.0)
    if not isinstance(colors, (list, np.ndarray)):
        if isinstance(alphas, (list, tuple, np.ndarray)):
            colors = []
            for i in range(num):
                colors.append(gl_color(colors, alphas[i]))
        else:
            if isinstance(colors, GLColor):
                colors = gl_color(colors, alphas)
            colors = [colors for i in range(num)]
    return np.array(colors).reshape([num, 4])


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

def _glcolor_to_qtcolor(glcolor):
    color = np.array(gl_color(glcolor)) * 255
    return QtGui.QColor(*color.astype(np.int32)) 

def get_rotation_matrix_3d(angle, axis=0):
    # points: [N, point_size, 3]
    # return counter-clockwise rotation matrix per axis
    rot_sin = np.sin(angle)
    rot_cos = np.cos(angle)
    if axis == 1:
        rot_mat_T = np.stack([[rot_cos, 0, -rot_sin], [0, 1, 0],
                              [rot_sin, 0, rot_cos]])
    elif axis == 2 or axis == -1:
        rot_mat_T = np.stack([[rot_cos, -rot_sin, 0], [rot_sin, rot_cos, 0],
                              [0, 0, 1]])
    elif axis == 0:
        rot_mat_T = np.stack([[0, rot_cos, -rot_sin], [0, rot_sin, rot_cos],
                              [1, 0, 0]])
    else:
        raise ValueError("axis should in range")
    return rot_mat_T.T



def get_C(fov, w, h):
    f = w / np.tan(fov / 2) / 2
    C = np.array([[f, 0, w / 2], [0, -f, h / 2], [0, 0, 1]], dtype=np.float64)
    return C


def get_RT(elevation, azimuth, distance, center):
    Ry = get_rotation_matrix_3d(-elevation, axis=1)
    Rz = get_rotation_matrix_3d(-(-azimuth + np.pi), axis=2)

    T0 = np.array([distance, 0, 0], dtype=np.float64)
    R = Rz @ Ry
    T = T0 - R @ np.array(center, dtype=np.float64)
    rect = np.array([[0, -1, 0], [0, 0, 1], [1, 0, 0]])
    return rect @ R, rect @ T


    # return R, T
def get_RT_(elevation, azimuth, distance, center):
    Ry = get_rotation_matrix_3d(-elevation, axis=1)
    Rz = get_rotation_matrix_3d(-(-azimuth + np.pi), axis=2)

    T0 = np.array([distance, 0, 0], dtype=np.float64)
    R = Rz @ Ry
    T = T0 - R @ np.array(center, dtype=np.float64)
    return R, T


def world2camera(points, elevation, azimuth, distance, center, fov, w, h):
    R, T = get_RT(elevation, azimuth, distance, center)
    C = get_C(fov, w, h)
    return C @ (R @ points + T)
    # return R @ points + T


def sph_to_device_pos(elevation, azimuth, distance, center):
    Ry = get_rotation_matrix_3d(elevation, axis=1)
    Rz = get_rotation_matrix_3d(-(azimuth + np.pi), axis=2)
    T0 = np.array([distance, 0, 0], dtype=np.float64)
    return Rz @ Ry @ T0 + Rz @ Ry @ np.array(center, dtype=np.float64)


def device_pos_to_sph(pos, center):
    pos -= center
    distance = np.linalg.norm(pos)
    elevation = np.arctan2(np.linalg.norm(pos[:2]), pos[2])
    azimuth = np.arctan2(pos[1], pos[0])
    return (elevation - np.pi / 2), (azimuth - np.pi), distance


def get_lines_for_circle(radius, num_point=100):
    rads = np.linspace(0, 2*np.pi, num=num_point)
    points = np.stack([radius * np.cos(rads), radius * np.sin(rads)], axis=1)
    lines = np.stack([points, points[[*np.arange(1, rads.shape[0]), 0]]], axis=1)
    return lines

class KittiGLViewWidget(gl.GLViewWidget):
    mousePressed = pyqtSignal(tuple, name="MousePressed")
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self._named_items = {}
        self.noRepeatKeys.append(QtCore.Qt.Key_W)
        self.noRepeatKeys.append(QtCore.Qt.Key_S)
        self.noRepeatKeys.append(QtCore.Qt.Key_A)
        self.noRepeatKeys.append(QtCore.Qt.Key_D)
        self.w_gl_axis = gl.GLAxisItem(
            size=None, antialias=True, glOptions='translucent')
        self.addItem(self.w_gl_axis)


    def scatter(self, name, points, colors=GLColor.Write, alphas=0.5,
                size=0.1, translucent=False):
        # if isinstance(colors, tuple):
        #     colors = gl_color(colors, alphas)
        colors = _extend_color_if_necessary(colors, points.shape[0], alphas)
        if name not in self._named_items:
            w_gl_scatter = gl.GLScatterPlotItem(
                pos=points, size=size, color=colors, pxMode=False)
            if translucent:
                w_gl_scatter.setGLOptions('translucent')
            self._named_items[name] = w_gl_scatter
            self.addItem(w_gl_scatter)
        else:
            self._named_items[name].setData(
                pos=points, size=size, color=colors, pxMode=False)

    def circles(self, name, poses, radiuses, colors=GLColor.Red, num_points=100, alphas=1.0,
                width=1.0, antialias=True):
        # if isinstance(colors, tuple):
        #     colors = gl_color(colors, alphas)
        
        if not isinstance(radiuses, (list, tuple, np.ndarray)):
            assert isinstance(poses, (list, tuple, np.ndarray))
            radiuses = np.full([len(poses)], radiuses)
        lines_list = []
        for pos, radius in zip(poses, radiuses):
            lines = get_lines_for_circle(radius, num_points)
            
            shape = [lines.shape[0], 2, 1]
            lines_with_z = np.concatenate([lines, np.full(shape, pos[-1])], axis=-1)
            lines_list.append(lines_with_z)
        lines = np.concatenate(lines_list, axis=0)
        return self.lines(name, lines.reshape(-1, 3), colors, alphas, width, antialias)


    def lines(self, name, lines, colors, alphas=1.0, width=1.0,
              antialias=True):
        if lines is None:
            return

        colors = _extend_color_if_necessary(colors, lines.shape[0], alphas)
        if name not in self._named_items:
            w_gl_item = gl.GLLinePlotItem(
                pos=lines,
                color=colors,
                width=width,
                antialias=antialias,
                mode='lines')
            self._named_items[name] = w_gl_item
            self.addItem(w_gl_item)
        else:
            self._named_items[name].setData(
                pos=lines,
                color=colors,
                width=width,
                antialias=antialias,
                mode='lines')

    def surface(self, name, x, y, z, colors, alphas=1.0):
        colors = _extend_color_if_necessary(colors, list(z.shape), alphas)
        if name not in self._named_items:
            w_gl_item = gl.GLSurfacePlotItem(
                x=x, y=y, z=z, shader=None, colors=colors, glOptions='translucent')
            self._named_items[name] = w_gl_item
            self.addItem(w_gl_item)
        else:
            self._named_items[name].setData(
                x=x, y=y, z=z, colors=colors)

    def labels(self, name, pos, labels, color, size=None, alpha=1.0):
        font = QtGui.QFont()
        if size is not None:
            font.setPointSizeF(size)
        if isinstance(color, GLColor):
            color = _glcolor_to_qtcolor(color)
        if isinstance(color, str):
            color = _pltcolor_to_qtcolor(color)

        if name not in self._named_items:
            w_gl_item = GLLabelItem(pos=pos, text=labels, color=color, font=font)
            self._named_items[name] = w_gl_item
            w_gl_item.setGLViewWidget(self)
            self.addItem(w_gl_item)
        else:
            self._named_items[name].setData(pos=pos, text=labels, color=color, font=font)

    def remove(self, name):
        if name in self._named_items:
            try:
                self.removeItem(self._named_items[name])
                return self._named_items.pop(name)
            except:
                print("remove failed.")

    def boxes3d(self,
                name,
                boxes,
                colors,
                width=1.0,
                alpha=1.0):
        if boxes.shape[0] == 0:
            self.remove(name)
            return
            # boxes = np.zeros([0, 8, 3])
        colors = _extend_color_if_necessary(colors, boxes.shape[0], alpha)
        total_lines = []
        total_colors = []
        for box, facecolor in zip(boxes, colors):
            lines = np.array([
                box[0], box[1], box[1], box[2], box[2], box[3], box[3], box[0],
                box[1], box[5], box[5], box[4], box[4], box[0], box[2], box[6],
                box[6], box[7], box[7], box[3], box[5], box[6], box[4], box[7]
            ])
            total_lines.append(lines)
            color = np.array([list(facecolor) for i in range(len(lines))])
            total_colors.append(color)
        if boxes.shape[0] != 0:
            total_lines = np.concatenate(total_lines, axis=0)
            total_colors = np.concatenate(total_colors, axis=0)
        else:
            total_lines = None
            total_colors = None
        self.lines(name, total_lines, total_colors, alphas=alpha, width=width)

    def set_camera_position(self,
                            center=None,
                            distance=None,
                            elevation=None,
                            azimuth=None):
        if center is not None:
            self.opts['center'].setX(center[0])
            self.opts['center'].setY(center[1])
            self.opts['center'].setZ(center[2])
        if distance is not None:
            self.opts['distance'] = distance
        if elevation is not None:
            self.opts['elevation'] = elevation
        if azimuth is not None:
            self.opts['azimuth'] = azimuth
        self.update()

    def reset_camera(self):
        self.set_camera_position(
            center=(5, 0, 0), distance=20, azimuth=-180, elevation=30)
        self.update()

    def evalKeyState(self):
        speed = 2.0
        world_speed = 0.32
        if len(self.keysPressed) > 0:
            for key in self.keysPressed:
                if key == QtCore.Qt.Key_Right:
                    self.orbit(azim=-speed, elev=0)
                elif key == QtCore.Qt.Key_Left:
                    self.orbit(azim=speed, elev=0)
                elif key == QtCore.Qt.Key_Up:
                    self.orbit(azim=0, elev=-speed)
                elif key == QtCore.Qt.Key_Down:
                    self.orbit(azim=0, elev=speed)
                elif key == QtCore.Qt.Key_W:
                    self.camera_move(dx=world_speed)
                elif key == QtCore.Qt.Key_S:
                    self.camera_move(dx=-world_speed)
                elif key == QtCore.Qt.Key_A:
                    self.camera_move(dy=world_speed)
                elif key == QtCore.Qt.Key_D:
                    self.camera_move(dy=-world_speed)
                elif key == QtCore.Qt.Key_Backspace:
                    self.reset_camera()
                elif key == QtCore.Qt.Key_PageUp:
                    pass
                elif key == QtCore.Qt.Key_PageDown:
                    pass
                self.keyTimer.start(16)
        else:
            self.keyTimer.stop()

    @property
    def camera_center(self):
        c = self.opts["center"]
        center = [c.x(), c.y(), c.z()]
        return np.array(center)

    @property
    def camera_position(self):
        c = self.opts["center"]
        center = [c.x(), c.y(), c.z()]
        elevation = self.opts['elevation'] / 180 * np.pi
        azimuth = self.opts['azimuth'] / 180 * np.pi
        distance = self.opts['distance']

        camera_pos = sph_to_device_pos(elevation, azimuth, distance, center)
        return camera_pos

    @property
    def camera_vector(self):
        vec = self.camera_position - self.camera_center
        return vec / np.linalg.norm(vec)

    def set_camera_position_cartesian(self, camera_pos, center):
        elevation, azimuth, distance = device_pos_to_sph(camera_pos, center)
        elevation = elevation * 180 / np.pi
        azimuth = azimuth * 180 / np.pi
        self.setCameraPosition(center, distance, elevation, azimuth)

    def camera_move(self, dx=0, dy=0, dz=0):
        xp = [dx, 0, 0]
        yp = [0, dy, 0]
        zp = [0, 0, dz]
        c = self.opts["center"]
        center = [c.x(), c.y(), c.z()]
        elevation = self.opts['elevation'] / 180 * np.pi
        azimuth = self.opts['azimuth'] / 180 * np.pi
        distance = self.opts['distance']
        R, T = get_RT_(elevation, azimuth, distance, center)
        dx_world = np.linalg.inv(R) @ (xp)
        dy_world = np.linalg.inv(R) @ (yp)
        dz_world = np.linalg.inv(R) @ (zp)
        # print(dx_world)
        return self.world_move(*(dx_world + dy_world + dz_world))

    def world_move(self, dx=0, dy=0, dz=0):
        c = self.opts["center"]
        center_new = [c.x() + dx, c.y() + dy, c.z() + dz]
        self.setCameraPosition(center=center_new)

    def mousePressEvent(self, ev):
        super().mousePressEvent(ev)
        self.mousePressed.emit((ev.x(), ev.y()))
        # print(ev.x(), ev.y())
    def mouseReleaseEvent(self, ev):

        # Example item selection code:
        # region = (ev.pos().x() - 5, ev.pos().y() - 5, 10, 10)
        # print(self.itemsAt(region))
        c = self.opts["center"]
        camera_info = (f"fov={self.opts['fov']:.2f}\n"
                       f"center=[{c.x():.2f}, {c.y():.2f}, {c.z():.2f}]\n"
                       f"distance={self.opts['distance']:.2f}\n"
                       f"elevation={self.opts['elevation']:.2f}\n"
                       f"azimuth={self.opts['azimuth']:.2f}")
        # print(camera_info)
        # w_norm = self.width() / self.height()
        # h_norm = 1.0
        w_norm = self.width()
        h_norm = self.height()
        elevation = self.opts['elevation'] / 180 * np.pi
        azimuth = self.opts['azimuth'] / 180 * np.pi
        distance = self.opts['distance']
        center = [c.x(), c.y(), c.z()]
        # center = [0, 0, 0]
        fov = self.opts['fov'] / 180 * np.pi
        # point = [12.35, 1.89, -1.82]
        point = [0, 0, 0.]
        ret = world2camera(
            np.array(point), elevation, azimuth, distance, center, fov, w_norm,
            h_norm)
        # print(ret[:2] / ret[2])
        # print(ret)
        # print(self.opts['center'])
        # print(self.opts['elevation'])
        # print(self.opts['azimuth'])
        # print(self.opts['distance'])
        pos = sph_to_device_pos(elevation, azimuth, distance, center)
        # print("camera pos:", pos)
        # print("camera pos inv", device_pos_to_sph(pos, center))
        ## debugging code: draw the picking region
        #glViewport(*self.getViewport())
        #glClear( GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT )
        #region = (region[0], self.height()-(region[1]+region[3]), region[2], region[3])
        #self.paintGL(region=region)
        #self.swapBuffers()