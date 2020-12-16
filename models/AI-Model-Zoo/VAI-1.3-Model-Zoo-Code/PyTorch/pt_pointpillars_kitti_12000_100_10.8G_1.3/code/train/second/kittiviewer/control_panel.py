
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

import contextlib
import enum
from functools import partial
import json
import numpy as np
from PyQt5 import QtCore, QtGui
from PyQt5.QtCore import QTimer, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QIcon, QMouseEvent, QPainter, QColor
from PyQt5.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDialog, QDoubleSpinBox, QFormLayout,
    QGroupBox, QHBoxLayout, QLabel, QLineEdit, QMainWindow, QPlainTextEdit,
    QPushButton, QSizePolicy, QSlider, QTabWidget, QVBoxLayout, QWidget, QColorDialog)


class Message:
    def __init__(self, name, value):
        self.name = name
        self.value = value


class QFloatSlider(QSlider):
    def __init__(self,
                 start,
                 stop,
                 num,
                 default_value=0,
                 orientation=QtCore.Qt.Horizontal,
                 parent=None):
        super().__init__(orientation, parent=parent)
        self._start = start
        self._stop = stop
        self._num = num
        self.setRange(0, num)
        self.set_value(default_value)

    def get_value(self, pos=None):
        if pos is None:
            pos = self.value()
        return (self._start - self._stop) / self._num * pos

    def set_value(self, val):
        pos = int(val * self._num / (self._start - self._stop))
        self.setSliderPosition(pos)

def _list_to_string(list_):
    return ", ".join([str(l) for l in list_])

def _string_to_list(string, dtype):
    str_list_ = string.replace(" ", "").split(',')
    return [dtype(l) for l in str_list_]

def _ndarray_to_string(array):
    return json.dumps(array.tolist())

def _string_to_ndarray(string, dtype):
    return np.array(json.loads(string), dtype=dtype)

def _shape_check(array, shape):
    if len(array.shape) != len(shape):
        raise ValueError("array must have same dim with shape")
    for s, sref in zip(array.shape, shape):
        if sref is not None and s != sref:
            raise ValueError("array must have same shape with shape")

def _jsonable_to_string(jsonable):
    return json.dumps(jsonable)

def _string_to_jsonable(string):
    return json.loads(string)

class QListEdit(QLineEdit):
    """ line edit for A, B, C data
    """
    def __init__(self,
                 dtype,
                 value=[],
                 parent=None):
        super().__init__(_list_to_string(value), parent=parent)
        self._dtype = dtype

    def get_value(self):
        return _string_to_list(self.text(), self._dtype)

    def set_value(self, value):
        self.setText(_list_to_string(value))

class QArrayEdit(QLineEdit):
    """ line edit for [[1, 2, 3]] data
    """
    def __init__(self,
                 dtype,
                 value=np.array([]),
                 shape=None,
                 parent=None):
        value = np.array(value, dtype=dtype)
        if shape is not None:
            _shape_check(value, shape)
        super().__init__(_ndarray_to_string(value), parent=parent)
        self._dtype = dtype
        self._shape = shape

    def get_value(self):
        return _string_to_ndarray(self.text(), self._dtype)

    def set_value(self, value):
        value = np.array(value, dtype=self._dtype)
        if self._shape is not None:
            _shape_check(value, self._shape)
        self.setText(_ndarray_to_string(value))

class QJsonEdit(QLineEdit):
    """ line edit for any json string data
    """
    def __init__(self,
                 value=[],
                 parent=None):
        super().__init__(_jsonable_to_string(value), parent=parent)

    def get_value(self):
        return _string_to_jsonable(self.text())

    def set_value(self, value):
        self.setText(_jsonable_to_string(value))


class QColorButton(QPushButton):
    '''
    Custom Qt Widget to show a chosen color.

    Left-clicking the button shows the color-chooser, while
    right-clicking resets the color to None (no-color).    
    '''

    colorChanged = pyqtSignal()

    def __init__(self, default_color=None, *args, **kwargs):
        super(QColorButton, self).__init__(*args, **kwargs)
        self._color = None
        if default_color is not None:
            self.gl_color = default_color
        self.setMaximumWidth(32)
        self.setColor(self._color)
        self.pressed.connect(self.onColorPicker)

    def setColor(self, color):
        if color != self._color:
            self._color = color
            self.colorChanged.emit()

        if self._color is not None:
            self.setStyleSheet(f"background-color: {QColor(self._color).name()};")
        else:
            self.setStyleSheet("")

    @property
    def rgba(self):
        return self._color
    
    @rgba.setter
    def rgba(self, value):
        self._color = value

    @property
    def gl_color(self):
        rgba = self._color
        alpha = ((rgba >> 24) & 0xff) / 255
        r = ((rgba >> 16) & 0xff) / 255
        g = ((rgba >> 8) & 0xff) / 255
        b = ((rgba) & 0xff) / 255
        return (r, g, b, alpha)

    @gl_color.setter
    def gl_color(self, color_tuple):
        r, g, b, alpha = (np.array(color_tuple) * 255).astype(np.int64)
        # rgba = alpha << 24 | r << 16 | g << 8 | b
        self._color = QColor(r, g, b, alpha).rgba()

    def onColorPicker(self):
        '''
        Show color-picker dialog to select color.

        Qt will use the native dialog by default.

        '''
        dlg = QColorDialog(self)
        if self._color:
            dlg.setCurrentColor(QColor(self._color))

        if dlg.exec_():
            # self.setColor(dlg.currentColor().name())
            self.setColor(dlg.currentColor().rgba())

class WidgetType(enum.Enum):
    Slider = "Slider"
    DoubleSpinBox = "DoubleSpinBox"
    LineEdit = "LineEdit"
    ComboBox = "ComboBox"
    CheckBox = "CheckBox"
    ListEdit = "ListEdit"
    ArrayEdit = "ArrayEdit"
    ColorButton = "ColorButton"
    JsonEdit = "JsonEdit"


class ControlPanel(QWidget):
    configChanged = pyqtSignal(Message, name="ConfigChanged")
    def __init__(self, column_nums=1, tab_num=1, parent=None):
        super().__init__(parent=parent)
        if not isinstance(column_nums, (list, tuple)):
            column_nums = [column_nums] * tab_num
        # self._layout = QHBoxLayout()
        self._tab = QTabWidget(self)
        # self._tab.resize(500, 500)
        self._tab_layouts = []
        for j in range(tab_num):
            from_layouts = []
            layout = QHBoxLayout()
            assert column_nums[j] > 0
            for i in range(column_nums[j]):
                f_layout = QFormLayout()
                from_layouts.append(f_layout)
                layout.addLayout(f_layout)
            w = QWidget(self._tab)
            w.setLayout(layout)
            self._tab.addTab(w, str(j))
            self._tab_layouts.append(from_layouts)
        self._widget_dict = {}
        self._widget_type = {}
        
        self._current_col = 0
        self._current_tab = 0
        self._column_nums = column_nums
        self._tab_num = tab_num
        # Add tabs
        self._layout = QVBoxLayout(self)
        self._layout.addWidget(self._tab)
        self.setLayout(self._layout)

    @contextlib.contextmanager
    def column(self, col):
        if col > (self._column_nums[self._current_tab] - 1):
            raise RuntimeError("column out of range")
        column_bkp = self._current_col
        self._current_col = col
        yield
        self._current_col = column_bkp

    @contextlib.contextmanager
    def tab(self, idx, name=None):
        if idx > (self._tab_num - 1):
            raise RuntimeError("tab index out of range")
        tab_idx_bkp = self._current_tab
        self._current_tab = idx
        if name is not None:
            self._tab.setTabText(idx, name)
        yield
        self._current_tab = tab_idx_bkp


    def add_slider(self, name, start, stop, num, default_value=0):
        ctrl = QFloatSlider(start, stop, num, default_value, parent=self)
        self._widget_dict[name] = ctrl
        slider_name = f"{name}({start:.2f}~{stop:.2f})"
        self._tab_layouts[self._current_tab][self._current_col].addRow(QLabel(slider_name), ctrl)
        ctrl.valueChanged.connect(
            partial(self.on_slider_valuechange, name=name))
        self._widget_type[name] = WidgetType.Slider

    def add_fspinbox(self, name, start, stop, step, default_value=None):
        ctrl = QDoubleSpinBox(self)
        ctrl.setRange(start, stop)
        ctrl.setSingleStep(step)
        if default_value is not None:
            ctrl.setValue(default_value)
        self._widget_dict[name] = ctrl
        slider_name = f"{name}({start:.2f}~{stop:.2f})"
        self._tab_layouts[self._current_tab][self._current_col].addRow(QLabel(slider_name), ctrl)
        ctrl.valueChanged.connect(
            partial(self.on_spinbox_valuechange, name=name))
        self._widget_type[name] = WidgetType.DoubleSpinBox

    def add_colorbutton(self, name, default_color=None):
        ctrl = QColorButton(default_color, parent=self)
        self._widget_dict[name] = ctrl
        self._tab_layouts[self._current_tab][self._current_col].addRow(QLabel(name), ctrl)
        ctrl.colorChanged.connect(
            partial(self.on_colorbutton_returned, name=name))
        self._widget_type[name] = WidgetType.ColorButton


    def add_lineedit(self, name):
        ctrl = QLineEdit(parent=self)
        self._widget_dict[name] = ctrl
        self._tab_layouts[self._current_tab][self._current_col].addRow(QLabel(name), ctrl)
        ctrl.returnPressed.connect(
            partial(self.on_lineedit_returnpressed, name=name))
        self._widget_type[name] = WidgetType.LineEdit

    def add_listedit(self, name, dtype, value=[]):
        ctrl = QListEdit(dtype, value, parent=self)
        self._widget_dict[name] = ctrl
        self._tab_layouts[self._current_tab][self._current_col].addRow(QLabel(name), ctrl)
        ctrl.returnPressed.connect(
            partial(self.on_listedit_returnpressed, name=name))
        self._widget_type[name] = WidgetType.ListEdit

    def add_arrayedit(self, name, dtype, value, shape=None):
        ctrl = QArrayEdit(dtype, value, shape, parent=self)
        self._widget_dict[name] = ctrl
        self._tab_layouts[self._current_tab][self._current_col].addRow(QLabel(name), ctrl)
        ctrl.returnPressed.connect(
            partial(self.on_arrayedit_returnpressed, name=name))
        self._widget_type[name] = WidgetType.ArrayEdit

    def add_jsonedit(self, name, value=[]):
        ctrl = QJsonEdit(value, parent=self)
        self._widget_dict[name] = ctrl
        self._tab_layouts[self._current_tab][self._current_col].addRow(QLabel(name), ctrl)
        ctrl.returnPressed.connect(
            partial(self.on_jsonedit_returnpressed, name=name))
        self._widget_type[name] = WidgetType.JsonEdit

    def add_checkbox(self, name):
        ctrl = QCheckBox(name, parent=self)
        self._widget_dict[name] = ctrl
        self._tab_layouts[self._current_tab][self._current_col].addRow(QLabel(name), ctrl)
        ctrl.stateChanged.connect(
            partial(self.on_checkbox_statechanged, name=name))
        self._widget_type[name] = WidgetType.CheckBox

    def add_combobox(self, name, values):
        ctrl = QComboBox(parent=self)
        for value in values:
            ctrl.addItem(value)
        self._widget_dict[name] = ctrl
        self._tab_layouts[self._current_tab][self._current_col].addRow(QLabel(name), ctrl)
        ctrl.currentTextChanged.connect(
            partial(self.on_combobox_changed, name=name))
        self._widget_type[name] = WidgetType.ComboBox

    def on_slider_valuechange(self, value, name):
        value = self._widget_dict[name].get_value(value)
        self.configChanged.emit(Message(name, value))

    def on_spinbox_valuechange(self, value, name):
        value = self._widget_dict[name].value()
        self.configChanged.emit(Message(name, value))

    def on_colorbutton_returned(self, name):
        color = self._widget_dict[name].gl_color
        self.configChanged.emit(Message(name, color))

    def on_lineedit_returnpressed(self, name):
        text = self._widget_dict[name].text()
        self.configChanged.emit(Message(name, text))

    def on_listedit_returnpressed(self, name):
        value = self._widget_dict[name].get_value()
        self.configChanged.emit(Message(name, value))

    def on_arrayedit_returnpressed(self, name):
        value = self._widget_dict[name].get_value()
        self.configChanged.emit(Message(name, value))

    def on_jsonedit_returnpressed(self, name):
        value = self._widget_dict[name].get_value()
        self.configChanged.emit(Message(name, value))

    def on_checkbox_statechanged(self, state, name):
        self.configChanged.emit(Message(name, state))

    def on_combobox_changed(self, name):
        value = self._widget_dict[name].currentText()
        self.configChanged.emit(Message(name, value))

    def set(self, name, value):
        if self._widget_type[name] == WidgetType.Slider:
            self._widget_dict[name].set_value(value)
        elif self._widget_type[name] == WidgetType.DoubleSpinBox:
            self._widget_dict[name].setValue(value)
        elif self._widget_type[name] == WidgetType.CheckBox:
            self._widget_dict[name].setChecked(value > 0)
        elif self._widget_type[name] == WidgetType.LineEdit:
            assert isinstance(value, str)
            self._widget_dict[name].setText(value)
        elif self._widget_type[name] == WidgetType.ComboBox:
            self._widget_dict[name].setCurrentIndex(value)
        elif self._widget_type[name] == WidgetType.ListEdit:
            self._widget_dict[name].set_value(value)
        elif self._widget_type[name] == WidgetType.ArrayEdit:
            self._widget_dict[name].set_value(value)
        elif self._widget_type[name] == WidgetType.JsonEdit:
            self._widget_dict[name].set_value(value)
        elif self._widget_type[name] == WidgetType.ColorButton:
            self._widget_dict[name].gl_color = value
        else:
            raise ValueError("unknown widget type")

    def get(self, name, save=False):
        if self._widget_type[name] == WidgetType.Slider:
            return self._widget_dict[name].get_value()
        elif self._widget_type[name] == WidgetType.DoubleSpinBox:
            return self._widget_dict[name].value()
        elif self._widget_type[name] == WidgetType.CheckBox:
            return self._widget_dict[name].isChecked()
        elif self._widget_type[name] == WidgetType.LineEdit:
            return self._widget_dict[name].text()
        elif self._widget_type[name] == WidgetType.ComboBox:
            if save:
                return self._widget_dict[name].currentIndex()
            else:
                return self._widget_dict[name].currentText()
        if self._widget_type[name] == WidgetType.ListEdit:
            return self._widget_dict[name].get_value()
        if self._widget_type[name] == WidgetType.ArrayEdit:
            return self._widget_dict[name].get_value()
        if self._widget_type[name] == WidgetType.JsonEdit:
            return self._widget_dict[name].get_value()
        if self._widget_type[name] == WidgetType.ColorButton:
            return self._widget_dict[name].gl_color
        else:
            raise ValueError("unknown widget type")

    def dumps(self):
        value_dict = {}
        for n, w in self._widget_dict.items():
            value_dict[n] = self.get(n, True)
            if isinstance(value_dict[n], np.ndarray):
                value_dict[n] = value_dict[n].tolist()
        return json.dumps(value_dict)
    
    def loads(self, string):
        value_dict = json.loads(string)
        for n, w in self._widget_dict.items():
            if n in value_dict:
                self.set(n, value_dict[n])


