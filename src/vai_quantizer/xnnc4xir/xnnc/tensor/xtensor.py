"""
 Copyright 2019 Xilinx Inc.

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

import logging
from enum import Enum
from typing import List, NoReturn, Optional, Tuple, Union

import numpy as np
from xnnc.proto.openir import tensor_pb2 as open_tensor

# create logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
# create console handler and set level to DEBUG
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
# create formatter
formatter = logging.Formatter("%(name)s - %(lineno)d - %(levelname)s - %(message)s")
# add formatter to ch
ch.setFormatter(formatter)
# add ch to logger
logger.addHandler(ch)

__all__ = ["XTensor", "XTensorCompute"]


class DataFormat(Enum):
    NCHW = 1
    NHWC = 2


class XTensorBase(object):
    """Xtensor base class"""

    def __init__(self, tensor: np.ndarray, format: Optional[DataFormat] = None):
        assert tensor is not None, "'tensor' should not be None."
        assert isinstance(
            tensor, np.ndarray
        ), "'tensor' should be a Numpy ndarray instance."
        if format:
            assert isinstance(
                format, DataFormat
            ), "'format' should be a DataFormat enum value."

        # numpy ndarray
        self.__ndarray: np.ndarray = tensor
        # data format: "NCHW" or "NHWC"
        self.__data_format: Optional[DataFormat] = format

    @property
    def dtype(self) -> np.dtype:
        return self.__ndarray.dtype

    @property
    def dtype_str(self) -> str:
        return self.__ndarray.dtype.name

    @property
    def shape(self) -> List[int]:
        return list(self.__ndarray.shape)

    @property
    def ndims(self) -> int:
        return len(self.__ndarray.shape)

    @property
    def data_format(self) -> DataFormat:
        return self.__data_format

    @data_format.setter
    def data_format(self, data_format: DataFormat) -> NoReturn:
        assert data_format is not None, "'data_format' should not be None."
        assert isinstance(
            data_format, DataFormat
        ), "'data_format' should be a DataFormat value."
        self.__data_format = data_format

    def to_numpy(self) -> np.ndarray:
        return self.__ndarray

    def astype(self, dtype: np.dtype) -> "XTensorBase":
        assert dtype is not None, "'dtype' should not be None."
        # assert isinstance(dtype, np.dtype), "'dtype' should be a Numpy dtype."
        self.__ndarray = self.__ndarray.astype(dtype)
        return self

    def tolist(self) -> List[int]:
        return self.__ndarray.tolist()

    def tobytes(self) -> bytes:
        return self.__ndarray.tobytes()


class XTensor(XTensorBase):
    """XTensor"""

    def __init__(self, tensor: np.ndarray, format: Optional[DataFormat] = None):
        XTensorBase.__init__(self, tensor, format)

    def reshape(self, shape: Union[Tuple[int], List[int]]) -> "XTensor":
        assert shape is not None and (
            isinstance(shape, tuple) or isinstance(shape, list)
        ), "'shape' should be a tuple or list of positive integers."
        return XTensor(self.to_numpy().reshape(tuple(shape)), format=self.data_format)

    def transpose(self, shape: Union[Tuple[int], List[int]]) -> "XTensor":
        assert shape is not None and (
            isinstance(shape, tuple) or isinstance(shape, list)
        ), "'shape' should be a tuple or list of positive integers."
        return XTensor(self.to_numpy().transpose(tuple(shape)).copy())

    def copy(self) -> "XTensor":
        return XTensor(self.to_numpy().copy(), format=self.data_format)

    def serialize(self) -> open_tensor.TensorProto:
        raise NotImplementedError("xtensor")

    @staticmethod
    def deserialize(tensor: open_tensor.TensorProto) -> "XTensor":
        assert tensor is not None, "'tensor' should not be None."
        assert isinstance(
            tensor, open_tensor.TensorProto
        ), "'tensor' should be of open_tensor.TensorProto type."

        dtype_str = open_tensor.TensorProto.DataType.Name(tensor.dtype).lower()
        shape = tensor.shape.dim
        return XTensor(
            np.frombuffer(tensor.content, dtype=np.dtype(dtype_str)).reshape(shape)
        )

    @staticmethod
    def zeros(
        shape: Union[List[int], Tuple[int]],
        dtype=np.float32,
        format: Optional[DataFormat] = None,
    ) -> "XTensor":
        assert (
            shape is not None
            and len(shape) > 0
            and (isinstance(shape, tuple) or isinstance(shape, list))
        ), "'shape' should be a tuple or list of positive integers."

        if isinstance(shape, list):
            shape = tuple(shape)

        tensor = np.zeros(shape, dtype)
        return XTensor(tensor, format=format)

    @staticmethod
    def ones(
        shape: Union[List[int], Tuple[int]],
        dtype=np.float32,
        format: Optional[DataFormat] = None,
    ) -> "XTensor":
        assert (
            shape is not None
            and len(shape) > 0
            and (isinstance(shape, tuple) or isinstance(shape, list))
        ), "'shape' should be a tuple or list of positive integers."

        if isinstance(shape, list):
            shape = tuple(shape)

        tensor = np.ones(shape, dtype)
        return XTensor(tensor, format=format)

    @staticmethod
    def zero_like(
        tensor: Union[np.ndarray, "XTensor"], format: Optional[DataFormat] = None
    ) -> "XTensor":
        assert tensor is not None and (
            isinstance(tensor, np.ndarray) or isinstance(tensor, XTensor)
        ), f"'tensor' should be an object of numpy ndarray or XTensor."

        shape = tensor.shape
        dtype = tensor.dtype
        if format:
            return XTensor.zeros(shape, dtype, format=format)
        elif isinstance(tensor, np.ndarray):
            return XTensor.zeros(shape, dtype)
        else:
            return XTensor.zeros(shape, dtype, format=tensor.data_format)

    @staticmethod
    def one_like(
        tensor: Union[np.ndarray, "XTensor"], format: Optional[DataFormat] = None
    ) -> "XTensor":
        assert tensor is not None and (
            isinstance(tensor) or isinstance(tensor, XTensor)
        ), f"'tensor' should be an object of numpy ndarray or XTensor."

        shape = tensor.shape
        dtype = tensor.dtype
        if format:
            return XTensor.ones(shape, dtype, format=format)
        elif isinstance(tensor, np.ndarray):
            return XTensor.ones(shape, dtype)
        else:
            return XTensor.ones(shape, dtype, format=tensor.data_format)


class XTensorCompute(object):
    """XTensorCompute class"""

    @classmethod
    def elem_add(cls, tensor1: XTensor, tensor2: XTensor) -> XTensor:
        assert tensor1 is not None, "'tensor1' should not be None."
        assert isinstance(tensor1, XTensor), "'tensor1' should be an XTensor instance."
        assert tensor2 is not None, "'tensor2' should not be None."
        assert isinstance(tensor2, XTensor), "'tensor2' should be an XTensor instance."
        assert (
            tensor1.dtype == tensor2.dtype
        ), f"'tensor1' and 'tensor2' should have same data type: actual: tensor1.dtype={tensor1.dtype}, tensor2.dtype={tensor2.dtype}"
        assert (
            tensor1.data_format == tensor2.data_format
        ), f"'tensor1' and 'tensor2' should have same data format: actual: tensor1.data_format={tensor1.data_format}, tensor2.data_format={tensor2.data_format}"

        return XTensor(
            np.add(tensor1.to_numpy(), tensor2.to_numpy()), format=tensor1.data_format
        )

    @classmethod
    def elem_mul(cls, tensor1: XTensor, tensor2: XTensor) -> XTensor:
        assert tensor1 is not None, "'tensor1' should not be None."
        assert isinstance(tensor1, XTensor), "'tensor1' should be an XTensor instance."
        assert tensor2 is not None, "'tensor2' should not be None."
        assert isinstance(tensor2, XTensor), "'tensor2' should be an XTensor instance."
        assert (
            tensor1.data_format == tensor2.data_format
        ), f"'tensor1' and 'tensor2' should have same data format: actual: tensor1.data_format={tensor1.data_format}, tensor2.data_format={tensor2.data_format}"

        out_tensor = np.multiply(tensor1.to_numpy(), tensor2.to_numpy())
        return XTensor(out_tensor, format=tensor1.data_format)

    @classmethod
    def elem_sub(cls, tensor1: XTensor, tensor2: XTensor) -> XTensor:
        """Subtract arguments, element-wise."""
        assert tensor1 is not None, "'tensor1' should not be None."
        assert isinstance(tensor1, XTensor), "'tensor1' should be an XTensor instance."
        assert tensor2 is not None, "'tensor2' should not be None."
        assert isinstance(tensor2, XTensor), "'tensor2' should be an XTensor instance."
        assert (
            tensor1.dtype == tensor2.dtype
        ), f"'tensor1' and 'tensor2' should have same data type: actual: tensor1.dtype={tensor1.dtype}, tensor2.dtype={tensor2.dtype}"
        assert (
            tensor1.data_format == tensor2.data_format
        ), f"'tensor1' and 'tensor2' should have same data format: actual: tensor1.data_format={tensor1.data_format}, tensor2.data_format={tensor2.data_format}"

        return XTensor(
            np.subtract(tensor1.to_numpy(), tensor2.to_numpy()),
            format=tensor1.data_format,
        )

    @classmethod
    def sum(
        cls,
        tensor: XTensor,
        axis: Union[int, Tuple[int]] = None,
        keepdims: bool = False,
    ) -> XTensor:
        """Sum of array elements over a given axis.

        Parameters
        ----------
        tensor : XTensor
            XTensor instance to sum
        axis : Union[int, Tuple[int]], optional
            Axis or axes along which a sum is performed. The default, axis=None, will
            sum all of the elements of the input array. If axis is negative it counts
            from the last to the first axis. If axis is a tuple of ints, a sum is
            performed on all of the axes specified in the tuple instead of a single
            axis or all the axes as before.

        Returns
        -------
        XTensor
            an XTensor instance with the same shape as input with the specified axis removed.
        """
        assert tensor is not None, "'tensor' should not be None."
        assert isinstance(tensor, XTensor), "'tensor' should be an XTensor instance."

        return XTensor(
            np.sum(tensor.to_numpy(), axis=axis, keepdims=keepdims),
            format=tensor.data_format,
        )

    @classmethod
    def pad(
        cls,
        tensor: XTensor,
        pad_width: List[int],
        mode: str = "constant",
        constant_values: Optional[List[int]] = None,
    ) -> XTensor:
        if mode == "constant":
            assert (
                constant_values is not None
            ), "'constant_values' should not be None when mode is 'constant'."
            return XTensor(
                np.pad(
                    tensor.to_numpy(),
                    pad_width,
                    mode=mode,
                    constant_values=constant_values,
                ),
                format=tensor.data_format,
            )
        else:
            return XTensor(
                np.pad(tensor.to_numpy(), pad_width, mode=mode),
                format=tensor.data_format,
            )

    @classmethod
    def mean(
        cls,
        tensor: XTensor,
        axis: Optional[Union[int, List[int]]] = None,
        keepdims: bool = False,
    ) -> XTensor:
        assert tensor is not None, "'tensor' should not be None."
        assert isinstance(tensor, XTensor), "'tensor' should be an XTensor instance."

        return XTensor(
            np.mean(tensor.to_numpy(), axis=tuple(axis), keepdims=keepdims),
            format=tensor.data_format,
        )

    @classmethod
    def squeeze(cls, tensor: XTensor, axis: Optional[List[int]] = None) -> XTensor:
        assert tensor is not None, "'tensor' should not be None."
        assert isinstance(tensor, XTensor), "'tensor' should be an XTensor instance."

        return XTensor(
            np.squeeze(tensor.to_numpy(), axis=axis), format=tensor.data_format
        )

    @classmethod
    def reshape(cls, tensor: XTensor, shape: Union[Tuple[int], List[int]]) -> XTensor:
        assert tensor is not None and isinstance(
            tensor, XTensor
        ), "'tensor' should be an XTensor object."
        assert shape is not None and (
            isinstance(shape, tuple) or isinstance(shape, list)
        ), "'shape' should be a tuple or list of positive integers."

        out_tensor = np.reshape(tensor.to_numpy(), tuple(shape))
        return XTensor(out_tensor, format=tensor.data_format)

    @classmethod
    def slice(
        cls,
        tensor: XTensor,
        begin: List[int],
        end: List[int],
        step: Optional[List[int]] = None,
        begin_mask: int = 0,
        end_mask: int = 0,
        shrink_axis_mask: int = 0,
    ) -> XTensor:
        assert tensor is not None, "'tensor' should not be None."
        assert isinstance(tensor, XTensor), "'tensor' should be an XTensor instance."
        assert (
            begin is not None
            and isinstance(begin, list)
            and all([isinstance(x, int) for x in begin])
        ), "'begin' should be an integer."
        assert (
            end is not None
            and isinstance(end, list)
            and all([isinstance(x, int) for x in end])
        ), "'end' should be an integer."
        assert len(begin) == len(end), "'begin' and 'end' should be the same length."
        if step:
            assert isinstance(step, list) and all(
                [isinstance(x, int) for x in step]
            ), "'step' should be a list of integers."
        else:
            step = [1] * tensor.ndims
        assert (
            begin_mask is not None and isinstance(begin_mask, int) and begin_mask >= 0
        ), "'begin_mask' should be a positive integer."
        assert (
            end_mask is not None and isinstance(end_mask, int) and end_mask >= 0
        ), "'end_mask' should be a positive integer."
        assert (
            shrink_axis_mask is not None
            and isinstance(shrink_axis_mask, int)
            and shrink_axis_mask >= 0
        ), "'shrink_axis_mask' should be a positive integer."
        for i, s in enumerate(step):
            if (1 << i) & end_mask != 0:
                continue

            if s >= 0:
                assert (
                    begin[i] <= end[i]
                ), f"'begin' should not be greater than 'end': begin:{begin}, end:{end}."
            else:
                assert (
                    begin[i] > end[i]
                ), f"'begin' should be greater than 'end' when 'step' is negative: begin:{begin}, end:{end}."

        # If the ith bit of begin_mask is set, begin[i] is ignored
        # and the fullest possible range in that dimension is used
        # instead. end_mask works analogously, except with the end range.
        if begin_mask != 0:
            for i in range(len(begin)):
                if (1 << i) & begin_mask != 0:
                    begin[i] = 0
        if end_mask != 0:
            for i in range(len(end)):
                if (1 << i) & end_mask != 0:
                    end[i] = tensor.shape[i]

        indexes = [slice(None)] * tensor.ndims
        for i in range(tensor.ndims):
            indexes[i] = slice(begin[i], end[i], step[i])

        # If the ith bit of shrink_axis_mask is set, it implies that the ith
        # specification shrinks the dimensionality by 1, taking on the value
        # at index begin[i]. end[i] and strides[i] are ignored in this case.
        if shrink_axis_mask != 0:
            for i in range(len(begin)):
                if (1 << i) & shrink_axis_mask != 0:
                    indexes[i] = slice(begin[i], begin[i] + 1)

        out_tensor = tensor.to_numpy()[tuple(indexes)]
        return XTensor(out_tensor, format=tensor.data_format)

    @classmethod
    def transpose(
        cls, tensor: XTensor, axes: Optional[Union[Tuple[int], List[int]]] = None
    ) -> XTensor:
        assert tensor is not None, "'tensor' should not be None."
        assert isinstance(tensor, XTensor), "'tensor' should be an XTensor instance."
        if axes:
            assert isinstance(axes, tuple) or isinstance(axes, list)
        return XTensor(
            np.transpose(tensor.to_numpy(), axes=axes).copy(), format=tensor.data_format
        )

    @classmethod
    def stack(cls, tensors: List[XTensor], axis: int) -> XTensor:
        assert tensors is not None, "'tensors' should not be None."
        assert all(
            [isinstance(x, XTensor) for x in tensors]
        ), "'tensor' should be an XTensor instance."
        assert len(tensors) > 1, "'tensors' should have two or more elements."
        assert axis is not None, "'axis' should not be None."
        assert isinstance(axis, int), "'axis' should be an integer."

        ndarrays = []
        for t in tensors:
            ndarrays.append(t.to_numpy())

        if all([x.shape == (1,) for x in ndarrays]) and axis == 0:
            scalars = []
            for x in ndarrays:
                scalars.append(x.item())
            return XTensor(
                np.array(scalars, dtype=np.int32), format=tensors[0].data_format
            )
        else:
            return XTensor(np.stack(ndarrays, axis=axis), format=tensors[0].data_format)

    @classmethod
    def concat(cls, tensors: List[XTensor], axis: int) -> XTensor:
        assert tensors is not None, "'tensors' should not be None."
        assert all(
            [isinstance(x, XTensor) for x in tensors]
        ), "'tensor' should be an XTensor instance."
        assert len(tensors) > 1, "'tensors' should have two or more elements."
        assert axis is not None, "'axis' should not be None."
        assert isinstance(axis, int), "'axis' should be an integer."
        format = tensors[0].data_format
        assert all(
            [format == t.data_format for t in tensors]
        ), f"all tensors should have the same data format setting."

        ndarrays = []
        for t in tensors:
            ndarrays.append(t.to_numpy())
        return XTensor(
            np.concatenate(ndarrays, axis=axis), format=tensors[0].data_format
        )

    @classmethod
    def reduce_max(
        cls, tensor: XTensor, axis: Optional[List[int]] = None, keepdims: bool = False
    ) -> XTensor:
        assert tensor is not None, "'tensor' should not be None."
        assert isinstance(tensor, XTensor), "'tensor' should be an XTensor instance."

        return XTensor(
            np.amax(tensor.to_numpy(), axis=tuple(axis), keepdims=keepdims),
            format=tensor.data_format,
        )

    @classmethod
    def reduce_sum(
        cls, tensor: XTensor, axis: Optional[List[int]] = None, keepdims: bool = False
    ) -> XTensor:
        assert tensor is not None, "'tensor' should not be None."
        assert isinstance(tensor, XTensor), "'tensor' should be an XTensor instance."

        return XTensor(
            np.sum(tensor.to_numpy(), axis=tuple(axis), keepdims=keepdims),
            format=tensor.data_format,
        )

    @classmethod
    def reduce_prod(
        cls, tensor: XTensor, axis: Optional[List[int]] = None, keepdims: bool = False
    ) -> XTensor:
        assert tensor is not None, "'tensor' should not be None."
        assert isinstance(tensor, XTensor), "'tensor' should be an XTensor instance."

        in_tensor = tensor.to_numpy()
        out_tensor = np.prod(
            in_tensor, axis=tuple(axis), keepdims=keepdims, dtype=in_tensor.dtype
        )

        if not isinstance(out_tensor, np.ndarray):
            if out_tensor.__class__ in [np.int32, np.int64]:
                out_tensor = np.array([out_tensor])
            else:
                raise NotImplementedError(
                    f"[ERROR] Not supported data type: {out_tensor.__class__.__name__}"
                )

        return XTensor(out_tensor, format=tensor.data_format)

    @classmethod
    def elem_exp(cls, tensor: XTensor) -> XTensor:
        assert tensor is not None, "'tensor' should not be None."
        assert isinstance(tensor, XTensor), "'tensor' should be an XTensor instance."

        return XTensor(np.exp(tensor.to_numpy()), format=tensor.data_format)

    @classmethod
    def elem_true_divide(cls, tensor1: XTensor, tensor2: XTensor) -> XTensor:
        assert tensor1 is not None, "'tensor1' should not be None."
        assert isinstance(tensor1, XTensor), "'tensor1' should be an XTensor instance."
        assert tensor2 is not None, "'tensor2' should not be None."
        assert isinstance(tensor2, XTensor), "'tensor2' should be an XTensor instance."
        assert (
            tensor1.dtype == tensor2.dtype
        ), f"'tensor1' and 'tensor2' should have same data type: actual: tensor1.dtype={tensor1.dtype}, tensor2.dtype={tensor2.dtype}"

        return XTensor(
            np.true_divide(tensor1.to_numpy(), tensor2.to_numpy()),
            format=tensor1.data_format,
        )

    @classmethod
    def elem_max(cls, tensor1: XTensor, tensor2: XTensor) -> XTensor:
        assert tensor1 is not None, "'tensor1' should not be None."
        assert isinstance(tensor1, XTensor), "'tensor1' should be an XTensor instance."
        assert tensor2 is not None, "'tensor2' should not be None."
        assert isinstance(tensor2, XTensor), "'tensor2' should be an XTensor instance."
        assert (
            tensor1.dtype == tensor2.dtype
        ), f"'tensor1' and 'tensor2' should have same data type: actual: tensor1.dtype={tensor1.dtype}, tensor2.dtype={tensor2.dtype}"

        return XTensor(
            np.maximum(tensor1.to_numpy(), tensor2.to_numpy()),
            format=tensor1.data_format,
        )

    @classmethod
    def elem_min(cls, tensor1: XTensor, tensor2: XTensor) -> XTensor:
        assert tensor1 is not None, "'tensor1' should not be None."
        assert isinstance(tensor1, XTensor), "'tensor1' should be an XTensor instance."
        assert tensor2 is not None, "'tensor2' should not be None."
        assert isinstance(tensor2, XTensor), "'tensor2' should be an XTensor instance."
        assert (
            tensor1.dtype == tensor2.dtype
        ), f"'tensor1' and 'tensor2' should have same data type: actual: tensor1.dtype={tensor1.dtype}, tensor2.dtype={tensor2.dtype}"

        return XTensor(
            np.minimum(tensor1.to_numpy(), tensor2.to_numpy()),
            format=tensor1.data_format,
        )

    @classmethod
    def dot(cls, tensor1: XTensor, tensor2: XTensor) -> XTensor:
        assert tensor1 is not None, "'tensor1' should not be None."
        assert isinstance(tensor1, XTensor), "'tensor1' should be an XTensor instance."
        assert tensor2 is not None, "'tensor2' should not be None."
        assert isinstance(tensor2, XTensor), "'tensor2' should be an XTensor instance."
        assert (
            tensor1.dtype == tensor2.dtype
        ), f"'tensor1' and 'tensor2' should have same data type: actual: tensor1.dtype={tensor1.dtype}, tensor2.dtype={tensor2.dtype}"

        return XTensor(
            np.dot(tensor1.to_numpy(), tensor2.to_numpy()), format=tensor1.data_format
        )

    @classmethod
    def matmul(cls, tensor1: XTensor, tensor2: XTensor) -> XTensor:
        assert tensor1 is not None, "'tensor1' should not be None."
        assert isinstance(tensor1, XTensor), "'tensor1' should be an XTensor instance."
        assert tensor2 is not None, "'tensor2' should not be None."
        assert isinstance(tensor2, XTensor), "'tensor2' should be an XTensor instance."
        assert (
            tensor1.data_format == tensor2.data_format
        ), f"two inputs tensor should have the same data format: tensor1:{tensor1.data_format.name}, tensor2:{tensor2.data_format.name}."
        assert (
            tensor1.shape[-1] == tensor2.shape[-2]
        ), f"[ERROR] The last dimention of 'tensor1' should be equal to the last second dimension of 'tensor2' : tensor1: {tensor1.shape}, matrix2: {tensor2.shape}"

        return XTensor(
            np.matmul(tensor1.to_numpy(), tensor2.to_numpy()),
            format=tensor1.data_format,
        )

    @classmethod
    def flip(cls, tensor: XTensor, axis: List[int] = None) -> XTensor:
        """
        Reverse the order of elements in tensor along the given axis.The shape of the tensor is preserved, but the elements are reordered.

        Parameters
        ----------
        tensor : XTensor
            an XTensor object.
        axis : List[int], optional
            Axis or axes along which to flip over, , by default None. The default will flip over all of the axes of the input array.

        Returns
        -------
        XTensor
            an XTensor object.
        """
        """


        Returns
        -------
        XTensor
            a new XTensor object with the reordered elements.
        """
        assert tensor is not None, "'tensor' should not be None."
        assert isinstance(tensor, XTensor), "'tensor' should be an XTensor instance."

        return XTensor(
            np.flip(
                tensor.to_numpy(), axis=tuple(axis) if axis is not None else None
            ).copy(),
            format=tensor.data_format,
        )

    @classmethod
    def ravel(cls, tensor: XTensor) -> XTensor:
        """
        Return a contiguous flattened XTensor object.

        Parameters
        ----------
        tensor : XTensor
            an XTensor object.

        Returns
        -------
        XTensor
            an flattend XTensor object.
        """
        assert tensor is not None, "'tensor' should not be None."
        assert isinstance(tensor, XTensor), "'tensor' should be an XTensor instance."

        return XTensor(np.ravel(tensor.to_numpy()), format=tensor.data_format)

    @classmethod
    def argmax(cls, tensor: XTensor, axis: int) -> XTensor:
        """
        Return the indices of the maximum values along an axis.

        Parameters
        ----------
        tensor : XTensor
            an XTensor object, input tensor.
        axis : int
            the axis to reduce across.

        Returns
        -------
        XTensor
            an XTensor object of indices into the array.
        """
        assert tensor is not None, "'tensor' should not be None."
        assert isinstance(tensor, XTensor), "'tensor' should be an XTensor instance."
        assert axis is not None, "'axis' should not be None."
        assert isinstance(axis, int), "'axis' should be an integer."

        return XTensor(np.argmax(tensor.to_numpy(), axis=axis), format=tensor.data_format)

    @classmethod
    def elem_true_power(cls, tensor1: XTensor, tensor2: XTensor) -> XTensor:
        assert tensor1 is not None, "'tensor1' should not be None."
        assert isinstance(tensor1, XTensor), "'tensor1' should be an XTensor instance."
        assert tensor2 is not None, "'tensor2' should not be None."
        assert isinstance(tensor2, XTensor), "'tensor2' should be an XTensor instance."
        assert (
            tensor1.dtype == tensor2.dtype
        ), f"'tensor1' and 'tensor2' should have same data type: actual: tensor1.dtype={tensor1.dtype}, tensor2.dtype={tensor2.dtype}"

        return XTensor(
            np.power(tensor1.to_numpy(), tensor2.to_numpy()),
            format=tensor1.data_format,
        )
