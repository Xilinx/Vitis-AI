/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/python/eager/pywrap_tensor.h"

#include <stdlib.h>
#include <string.h>

#include "structmember.h"  // NOLINT // For PyMemberDef
#include "tensorflow/c/c_api.h"
#include "tensorflow/c/eager/c_api.h"
#include "tensorflow/c/eager/c_api_internal.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/python/eager/pywrap_tensor_conversion.h"
#include "tensorflow/python/eager/pywrap_tfe.h"
#include "tensorflow/python/lib/core/ndarray_tensor.h"
#include "tensorflow/python/lib/core/ndarray_tensor_bridge.h"
#include "tensorflow/python/lib/core/numpy.h"
#include "tensorflow/python/lib/core/py_seq_tensor.h"
#include "tensorflow/python/lib/core/safe_ptr.h"

// forward declare
struct EagerTensor;

namespace {

// An instance of _EagerTensorProfiler that will receive callbacks about
// events on eager tensors. This is set by TFE_Py_InitEagerTensor, if at all.
PyObject* eager_tensor_profiler = nullptr;

TFE_Context* GetContextHandle(PyObject* py_context) {
  tensorflow::Safe_PyObjectPtr py_context_handle(
      PyObject_GetAttrString(py_context, "_handle"));
  if (py_context_handle == nullptr) {
    // Current Python code makes sure this never happens. If it does, or
    // becomes hard to maintain, we can call the ensure_initialized() method
    // here.
    PyErr_SetString(
        PyExc_TypeError,
        "Expected `context` argument in EagerTensor constructor to have a "
        "`_handle` attribute but it did not. Was eager Context initialized?");
    return nullptr;
  }

  auto* ctx = reinterpret_cast<TFE_Context*>(
      PyCapsule_GetPointer(py_context_handle.get(), nullptr));
  if (ctx == nullptr) {
    PyErr_SetString(PyExc_TypeError,
                    tensorflow::strings::StrCat(
                        "Expected context._handle to contain a PyCapsule "
                        "encoded pointer to TFE_Context. Got ",
                        Py_TYPE(py_context_handle.get())->tp_name)
                        .c_str());
  }
  return ctx;
}

// Convert a Python numpy.ndarray object to a TFE_TensorHandle.
// The two may share underlying storage so changes to one may reflect in the
// other.
TFE_TensorHandle* NumpyToTFE_TensorHandle(PyObject* obj) {
  tensorflow::TensorHandle* handle;
  tensorflow::Tensor t;
  auto cppstatus = tensorflow::NdarrayToTensor(obj, &t);
  if (cppstatus.ok()) {
    cppstatus = tensorflow::TensorHandle::CreateLocalHandle(t, &handle);
  }
  if (!cppstatus.ok()) {
    PyErr_SetString(PyExc_ValueError,
                    tensorflow::strings::StrCat(
                        "Failed to convert a NumPy array to a Tensor (",
                        cppstatus.error_message(), ").")
                        .c_str());
    return nullptr;
  }
  return new TFE_TensorHandle(handle);
}

// Convert a TFE_TensorHandle to a Python numpy.ndarray object.
// The two may share underlying storage so changes to one may reflect in the
// other.
PyObject* TFE_TensorHandleToNumpy(TFE_TensorHandle* handle, TF_Status* status) {
  if (TFE_TensorHandleDataType(handle) == TF_RESOURCE) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 "Cannot convert a Tensor of dtype resource to a NumPy array.");
    return nullptr;
  }

  tensorflow::Safe_TF_TensorPtr tensor = nullptr;
  Py_BEGIN_ALLOW_THREADS;
  tensor = tensorflow::make_safe(TFE_TensorHandleResolve(handle, status));
  Py_END_ALLOW_THREADS;
  if (TF_GetCode(status) != TF_OK) {
    return nullptr;
  }

  PyObject* ret = nullptr;
  auto cppstatus =
      tensorflow::TF_TensorToMaybeAliasedPyArray(std::move(tensor), &ret);
  tensorflow::Set_TF_Status_from_Status(status, cppstatus);
  if (TF_GetCode(status) != TF_OK) {
    Py_XDECREF(ret);
    return nullptr;
  }
  CHECK_NE(ret, nullptr);
  return ret;
}

// Helper function to convert `v` to a tensorflow::DataType and store it in
// `*out`. Returns true on success, false otherwise.
// Note that we assume that v is a python int (not long) representing a
// TF_DataType/tensorflow::DataType value.
bool PyIntToDataType(PyObject* v, tensorflow::DataType* out) {
#if PY_MAJOR_VERSION < 3
  if (PyInt_Check(v)) {
    *out = static_cast<tensorflow::DataType>(PyInt_AS_LONG(v));
    return true;
  }
#else
  if (PyLong_Check(v)) {
    *out = static_cast<tensorflow::DataType>(PyLong_AsLong(v));
    return true;
  }
#endif
  return false;
}

// Helper function to create a python integer from TF_DataType.
PyObject* PyIntFromDataType(TF_DataType l) {
#if PY_MAJOR_VERSION < 3
  return PyInt_FromLong(l);
#else
  return PyLong_FromLong(l);
#endif
}

// PyObject->tensorflow::DataType conversion function to be used with
// PyArg_Parse* APIs.
int ConvertDataType(PyObject* obj, tensorflow::DataType* dst) {
  if (obj == Py_None) {
    *dst = tensorflow::DataType::DT_INVALID;
  } else if (!PyIntToDataType(obj, dst)) {
    PyErr_SetString(
        PyExc_TypeError,
        tensorflow::strings::StrCat(
            "Expecting a DataType value for dtype. Got ", Py_TYPE(obj)->tp_name)
            .c_str());
    return 0;
  }

  return 1;
}

// Conversion function extracting a const char** device name from a PyObject.
// The function should be used with PyArg_Parse* APIs.
int ConvertDeviceName(PyObject* obj, const char** dst) {
  if (obj == Py_None) {
    *dst = nullptr;
  } else {
    auto device_name = TFE_GetPythonString(obj);
    if (device_name == nullptr) {
      PyErr_Clear();
      PyErr_SetString(PyExc_TypeError, "Error parsing device argument.");
      return 0;
    }
    *dst = device_name;
  }

  return 1;
}

}  // namespace

namespace tensorflow {
// This function checks whether the desired type is "compatible" with the
// inferred type. At a high level, compatibility means that all integral types
// are compatible with each other, and all floating types are compatible with
// each other.
//
// Type compatibility doesn't consider overflows (i.e. int64 is *always*
// compatible with int32). This is intended to match graph behavior.
bool IsCompatible(DataType desired, DataType returned) {
  if (desired == returned) return true;

  if (DataTypeIsInteger(desired) && DataTypeIsInteger(returned)) {
    return true;
  } else if (DataTypeIsFloating(desired) &&
             (DataTypeIsFloating(returned) || DataTypeIsInteger(returned))) {
    return true;
  } else if (DataTypeIsComplex(desired) &&
             (DataTypeIsComplex(returned) || DataTypeIsInteger(returned) ||
              DataTypeIsFloating(returned))) {
    return true;
  } else if (DataTypeIsQuantized(desired) && DataTypeIsInteger(returned)) {
    return true;
  }
  return false;
}

// TODO(nareshmodi): Move EagerCast and ReadVariableOp (which use the C API to
// execute TFE Ops) to a separate common library.
// Casts data referred to by `handle` from type `src_type_enum` to type
// `dst_type_enum`.
TFE_TensorHandle* EagerCast(TFE_Context* ctx, TFE_TensorHandle* handle,
                            TF_DataType src_type_enum,
                            TF_DataType dst_type_enum, TF_Status* out_status) {
  if (ctx == nullptr) return nullptr;
  const char* op_name = "Cast";
  const char* device_name = "/device:CPU:0";
  TFE_Op* op = TFE_NewOp(ctx, op_name, out_status);
#define RETURN_ERROR  \
  {                   \
    TFE_DeleteOp(op); \
    return nullptr;   \
  }
  if (TF_GetCode(out_status) != TF_OK) RETURN_ERROR
  TFE_OpSetDevice(op, device_name, out_status);
  if (TF_GetCode(out_status) != TF_OK) RETURN_ERROR
  TFE_OpAddInput(op, handle, out_status);
  if (TF_GetCode(out_status) != TF_OK) RETURN_ERROR
  TFE_OpSetAttrType(op, "SrcT", src_type_enum);
  TFE_OpSetAttrType(op, "DstT", dst_type_enum);
  TFE_OpSetAttrBool(op, "Truncate", false);
  TFE_TensorHandle* output = nullptr;
  int num_outputs = 1;
  TFE_Execute(op, &output, &num_outputs, out_status);
  if (TF_GetCode(out_status) != TF_OK || num_outputs != 1 ||
      output == nullptr) {
    if (output != nullptr) {
      TFE_DeleteTensorHandle(output);
    }
    RETURN_ERROR
  }
  TFE_DeleteOp(op);
  return output;
#undef RETURN_ERROR
}

TFE_TensorHandle* PySeqToTFE_TensorHandle(PyObject* value, DataType dtype) {
  tensorflow::TensorHandle* handle = nullptr;
  tensorflow::Tensor t;
  // TODO(josh11b): Have PySeqToTensor set python errors instead of
  // returning Status.
  auto cppstatus = tensorflow::PySeqToTensor(value, dtype, &t);
  if (cppstatus.ok()) {
    cppstatus = tensorflow::TensorHandle::CreateLocalHandle(t, &handle);
  }
  if (!cppstatus.ok()) {
    PyErr_SetString(PyExc_ValueError, cppstatus.error_message().c_str());
    return nullptr;
  }
  CHECK_NE(handle, nullptr);
  return new TFE_TensorHandle(handle);
}

TFE_TensorHandle* ConvertToEagerTensorUncached(TFE_Context* ctx,
                                               PyObject* value,
                                               tensorflow::DataType dtype,
                                               const char* device_name) {
  tensorflow::Safe_PyObjectPtr value_decrefer;
  if (PyArray_IsScalar(value, Generic)) {
    // Convert numpy scalars to numpy arrays.
    value = PyArray_FromScalar(value, nullptr);
    // The returned value needs to be DECREF'd, but the original value was
    // created in python code, and doesn't need to be DECREF'd.
    value_decrefer.reset(value);
  }

  Safe_TFE_TensorHandlePtr handle;
  if (PyArray_Check(value)) {
    int desired_np_dtype = -1;
    if (dtype != tensorflow::DT_INVALID) {
      if (!tensorflow::TF_DataType_to_PyArray_TYPE(
               static_cast<TF_DataType>(dtype), &desired_np_dtype)
               .ok()) {
        PyErr_SetString(
            PyExc_TypeError,
            tensorflow::strings::StrCat("Invalid dtype argument value ", dtype)
                .c_str());
        return nullptr;
      }
    }
    PyArrayObject* array = reinterpret_cast<PyArrayObject*>(value);
    int current_np_dtype = PyArray_TYPE(array);
    auto safe_value = tensorflow::make_safe(static_cast<PyObject*>(nullptr));
    if ((desired_np_dtype >= 0 && desired_np_dtype != current_np_dtype) ||
        !PyArray_ISCARRAY(array)) {
      int new_dtype =
          desired_np_dtype >= 0 ? desired_np_dtype : current_np_dtype;
      safe_value = tensorflow::make_safe(
          PyArray_FromAny(value, PyArray_DescrFromType(new_dtype), 0, 0,
                          NPY_ARRAY_CARRAY_RO | NPY_ARRAY_FORCECAST, nullptr));
      if (PyErr_Occurred()) return nullptr;
      if (safe_value == nullptr) {
        PyErr_SetString(PyExc_ValueError, "Error while casting a numpy value");
        return nullptr;
      }
      value = safe_value.get();
    }
    handle = make_safe(NumpyToTFE_TensorHandle(value));
  } else {
    handle = make_safe(PySeqToTFE_TensorHandle(value, dtype));
  }

  if (handle == nullptr) return nullptr;

  Safe_TF_StatusPtr status = make_safe(TF_NewStatus());
  TF_DataType handle_dtype = TFE_TensorHandleDataType(handle.get());
  if (dtype != tensorflow::DT_INVALID &&
      dtype != static_cast<DataType>(handle_dtype)) {
    if (tensorflow::IsCompatible(dtype, static_cast<DataType>(handle_dtype))) {
      handle = tensorflow::make_safe(
          tensorflow::EagerCast(ctx, handle.get(), handle_dtype,
                                static_cast<TF_DataType>(dtype), status.get()));
      if (TF_GetCode(status.get()) != TF_OK) {
        PyErr_SetString(PyExc_TypeError,
                        absl::StrCat("Error while casting from dtype ",
                                     tensorflow::DataTypeString(
                                         static_cast<DataType>(handle_dtype)),
                                     " to ", tensorflow::DataTypeString(dtype),
                                     ". ", TF_Message(status.get()))
                            .c_str());
        return nullptr;
      }
    } else {
      tensorflow::Safe_PyObjectPtr value_str(PyObject_Repr(value));
      PyErr_SetString(
          PyExc_TypeError,
          absl::StrCat("Cannot convert ", TFE_GetPythonString(value_str.get()),
                       " to EagerTensor of dtype ",
                       tensorflow::DataTypeString(dtype))
              .c_str());
      return nullptr;
    }
  }

  // Almost all TensorFlow kernels for GPU devices keep int32 tensors in host
  // memory. We approximate the same behavior for eager execution - keeping
  // int32 tensors in host memory.
  //
  // We do so to preclude the need for callers into such kernels from having to
  // explicitly place the int32 tensors in host memory. For example, without
  // this, one needed:
  //
  // with tf.device('/gpu:0'):
  //   ...// code here
  //   with tf.device('/cpu:0'):
  //     shape = tf.constant(...)
  //   y = tf.random_uniform(shape)
  //
  // Without the CPU device block, tfe.ops.random_uniform would fail since the
  // kernel expects the shape in host memory.
  //
  // With this support, we simplify the code:
  //
  // with tf.device('/gpu:0'):
  //   y = tf.random_uniform(...)
  //
  // The approximation is not exact there are GPU kernels which do not require
  // host memory for int32 tensors. This will lead to a discrepancy between
  // eager and graph execution.
  //
  // To support remote execution copy int32 tensors to another CPU device.
  // TODO(ashankar): Fix this.
  if (device_name != nullptr &&
      (TFE_TensorHandleDataType(handle.get()) != TF_INT32 ||
       strstr(device_name, "/device:CPU:0") != nullptr)) {
    // Note that this is a shallow copy and will share the underlying buffer
    // if copying to the same device.
    handle = make_safe(TFE_TensorHandleCopyToDevice(handle.get(), ctx,
                                                    device_name, status.get()));
    if (MaybeRaiseExceptionFromTFStatus(status.get(), PyExc_RuntimeError)) {
      return nullptr;
    }
  }

  return handle.release();
}

TFE_TensorHandle* ConvertToEagerTensor(TFE_Context* ctx, PyObject* value,
                                       DataType dtype,
                                       const char* device_name) {
  // Reduce the overhead of allocation/transfer-to-device for scalars by
  // caching the corresponding handles. Note that currently only Python
  // scalars are cached.
  // TODO(slebedev): also cache singleton NumPy arrays and scalars?
  if (PyArray_IsPythonNumber(value)) {
    auto* cache = TFE_TensorHandleCache::Get();
    TFE_TensorHandle* handle = cache->Lookup(value, dtype, device_name);
    if (handle != nullptr) return handle;
    handle = ConvertToEagerTensorUncached(ctx, value, dtype, device_name);
    if (handle == nullptr) return nullptr;
    cache->Insert(value, dtype, device_name, handle);
    return handle;
  } else {
    return ConvertToEagerTensorUncached(ctx, value, dtype, device_name);
  }
}

}  // namespace tensorflow

extern "C" {

static const int kMaxEagerTensorParentSize = 64;

// TODO(agarwal): store context handle in EagerTensor.
typedef struct EagerTensor {
  PyObject_HEAD;
  // Note that we leave kMaxEagerTensorParentSize bytes here for use by the
  // parent class. The parent class is set at runtime, so we don't know the
  // exact size at compile time.
  char unused[kMaxEagerTensorParentSize];
  TFE_TensorHandle* handle;
  int64_t id;
  // This mirrors tensorflow.core.framework.ops.Tensor._handle_data Which will
  // be None for tensors of type other than DT_REOSURCE. For DT_RESOURCE
  // tensors, this will contain a serialized HandleData proto with shape
  // inference metadata about shapes and dtypes of resources accessible from
  // this handle.
  // Note that we assume that handle_data cannot participate in reference
  // cycles, and hence don't provide GC support for it.
  PyObject* handle_data;

  // This stores `_tensor_shape`, a cached `TensorShape` object, and is set the
  // first time that `_EagerTensorBase`'s `shape` property is called.
  PyObject* tensor_shape;

  // We store a status object here as an optimization to avoid allocating a new
  // Status objects on different functions that operate on EagerTensor and need
  // to use a TF_Status object. However note that accesses to `status` are not
  // thread-safe.
  TF_Status* status;

  // The eager Context (from eager/context.py) used by this Tensor.
  // This is currently used only to make sure context outlives TensorHandles.
  PyObject* context;

  PyObject* weakreflist; /* List of weak references */

  // Per-instance attribute dictionary, to support monkey patching
  // (e.g. EagerTensor.assign when slicing variables). This dictionary is
  // created by CPython the first time an attribute is assigned, pointed to by
  // tp_dictoffset. Note that garbage collection is not enabled for
  // EagerTensors, so assigning objects to EagerTensor attributes which require
  // garbage collection is likely to cause issues.
  PyObject* dict;
} EagerTensor;

namespace {

// Returns true on success - successfully invoked or no profiler registered.
// Returns false if some error occurred.
bool MaybeInvokeCreatedOnEagerTensorProfiler(EagerTensor* created_tensor) {
  if (eager_tensor_profiler != nullptr) {
#if PY_MAJOR_VERSION < 3
    PyObject* created_method_name = PyString_InternFromString("created");
#else
    PyObject* created_method_name = PyUnicode_InternFromString("created");
#endif
    if (created_method_name == nullptr) {
      return false;
    }
    PyObject* result = PyObject_CallMethodObjArgs(
        eager_tensor_profiler, created_method_name, created_tensor, NULL);
    if (result == nullptr) {
      LOG(ERROR) << "Invoking created() on EagerTensor profiler failed";
      // While we can potentially continue because the error is related to
      // profiling, we choose to return an error because:
      //  - If profiling is used, the user likely wants to stop execution on
      //    profiling errors.
      //  - Error in profiling code might have left some state in an invalid
      //    form that can lead to an error later on. Better to fail fast.
      Py_DECREF(created_method_name);
      return false;
    }
    Py_DECREF(created_method_name);
    Py_DECREF(result);
  }
  return true;
}

}  // namespace

// tp_init for EagerTensor.
int EagerTensor_init(EagerTensor* self, PyObject* args, PyObject* kwds) {
  self->id = get_uid();
  self->handle = nullptr;
  Py_INCREF(Py_None);
  self->handle_data = Py_None;
  Py_INCREF(Py_None);
  self->tensor_shape = Py_None;
  self->status = TF_NewStatus();
  self->dict = nullptr;
  self->weakreflist = nullptr;
  self->context = nullptr;
  PyObject* value;
  const char* device_name = nullptr;
  tensorflow::DataType dtype = tensorflow::DataType::DT_INVALID;
  const char* kwlist[] = {"value", "device", "dtype", nullptr};
  if (!PyArg_ParseTupleAndKeywords(
          args, kwds, "OO&|O&", const_cast<char**>(kwlist), &value,
          ConvertDeviceName, &device_name, ConvertDataType, &dtype)) {
    return -1;
  }

  PyObject* py_context = GetPyEagerContext();
  if (py_context == nullptr) return -1;
  self->context = py_context;

  auto* handle = tensorflow::ConvertToEagerTensor(GetContextHandle(py_context),
                                                  value, dtype, device_name);
  if (handle == nullptr) return -1;
  self->handle = handle;

  if (!MaybeInvokeCreatedOnEagerTensorProfiler(self)) {
    return -1;
  }

  return 0;
}

// tp_dealloc for EagerTensor.
void EagerTensor_dealloc(EagerTensor* self) {
  // Unhook the object from python's GC so that the weakref deleter doesn't
  // try to re-delete this.
  PyObject_GC_UnTrack((PyObject*)self);

  // Clear weak references to self.
  // Needs to happen before any actual destruction.
  PyObject_ClearWeakRefs((PyObject*)self);

  TF_DeleteStatus(self->status);
  Py_DECREF(self->handle_data);
  Py_DECREF(self->tensor_shape);
  // If an attribute dictionary has been created, release it. Note that this
  // is only ever created by CPython's attribute setting methods; we don't
  // create it ourselves.
  Py_CLEAR(self->dict);
  if (self->handle != nullptr) {
    TFE_DeleteTensorHandle(self->handle);
    self->handle = nullptr;
  }

  // Decref context after deleting the tensor handle.
  Py_XDECREF(self->context);

  // We have the global interpreter lock, so use this chance to perform delayed
  // refcount decrements.
  tensorflow::ClearDecrefCache();
  auto id = self->id;
  Py_TYPE(self)->tp_free(self);
  TFE_Py_TapeSetDeleteTrace(id);
}

// Getter for `_id`.
static PyObject* EagerTensor_getid(EagerTensor* self, void* closure) {
  return PyLong_FromLongLong(self->id);
}

// Getter for `_datatype_enum`.
static PyObject* EagerTensor_datatype_enum(EagerTensor* self) {
  return PyIntFromDataType(TFE_TensorHandleDataType(self->handle));
}

// Getter for `_shape_tuple`.
static PyObject* EagerTensor_shape_tuple(EagerTensor* self) {
  auto handle = self->handle;
  int n = TFE_TensorHandleNumDims(handle, self->status);
  if (MaybeRaiseExceptionFromTFStatus(self->status, PyExc_ValueError)) {
    // Cleanup self->status before returning.
    TF_SetStatus(self->status, TF_OK, "");
    return nullptr;
  }
  PyObject* shape = PyTuple_New(n);
  if (PyErr_Occurred()) return nullptr;
  for (int i = 0; i < n; ++i) {
    PyObject* dim =
        PyLong_FromLongLong(TFE_TensorHandleDim(handle, i, self->status));
    if (MaybeRaiseExceptionFromTFStatus(self->status, PyExc_ValueError) ||
        dim == nullptr || PyTuple_SetItem(shape, i, dim) != 0) {
      // Cleanup self->status before returning.
      TF_SetStatus(self->status, TF_OK, "");
      Py_DECREF(shape);
      if (dim != nullptr) Py_DECREF(dim);
      PyErr_SetString(PyExc_RuntimeError, "Error while creating shape");
      return nullptr;
    }
  }
  return shape;
}

// Getter for `_rank`.
static PyObject* EagerTensor_rank(EagerTensor* self) {
  int num_dims = TFE_TensorHandleNumDims(self->handle, self->status);
  if (MaybeRaiseExceptionFromTFStatus(self->status, PyExc_ValueError)) {
    // Cleanup self->status before returning.
    TF_SetStatus(self->status, TF_OK, "");
    return nullptr;
  }
#if PY_MAJOR_VERSION < 3
  return PyInt_FromLong(num_dims);
#else
  return PyLong_FromLong(num_dims);
#endif
}

// Getter for `_num_elements`.
static PyObject* EagerTensor_num_elements(EagerTensor* self) {
  auto handle = self->handle;
  int n = TFE_TensorHandleNumElements(handle, self->status);
  if (MaybeRaiseExceptionFromTFStatus(self->status, PyExc_ValueError)) {
    // Cleanup self->status before returning.
    TF_SetStatus(self->status, TF_OK, "");
    return nullptr;
  }
  return PyLong_FromLongLong(n);
}

static PyObject* EagerTensor_tensor_handle(EagerTensor* self, void* unused) {
  Py_INCREF(self->handle_data);
  return self->handle_data;
}

static int EagerTensor_settensor_handle(EagerTensor* self, PyObject* value,
                                        void* unused) {
  Py_DECREF(self->handle_data);
  Py_INCREF(value);
  self->handle_data = value;
  return 0;
}

static PyObject* EagerTensor_tensor_shape(EagerTensor* self, void* unused) {
  Py_INCREF(self->tensor_shape);
  return self->tensor_shape;
}

static int EagerTensor_settensor_shape(EagerTensor* self, PyObject* value,
                                       void* unused) {
  Py_DECREF(self->tensor_shape);
  Py_INCREF(value);
  self->tensor_shape = value;
  return 0;
}

// Function `_copy_to_device`.
static PyObject* EagerTensor_copy_to_device(EagerTensor* self, PyObject* args,
                                            PyObject* kwds) {
  if (!_PyArg_NoKeywords("copy_to_device", kwds)) return nullptr;

  const char* device_name = nullptr;
  if (!PyArg_ParseTuple(args, "O&:copy_to_device", ConvertDeviceName,
                        &device_name)) {
    return nullptr;
  }

  // Note that this is a shallow copy and will share the underlying buffer
  // if copying to the same device.
  TFE_TensorHandle* handle = TFE_TensorHandleCopyToDevice(
      self->handle, GetContextHandle(self->context), device_name, self->status);
  if (MaybeRaiseExceptionFromTFStatus(self->status, PyExc_RuntimeError)) {
    // Cleanup self->status before returning.
    TF_SetStatus(self->status, TF_OK, "");
    return nullptr;
  }

  return EagerTensorFromHandle(handle);
}

// Function `_numpy`.
// Convert an EagerTensor to a Python numpy.ndarray object.
// The two may share underlying storage so changes to one may reflect in the
// other.
// Note that if `self` is not on CPU, we raise an Exception.
static PyObject* EagerTensor_numpy(EagerTensor* self) {
  auto* py_array = TFE_TensorHandleToNumpy(self->handle, self->status);
  if (MaybeRaiseExceptionFromTFStatus(self->status, PyExc_ValueError)) {
    Py_XDECREF(py_array);
    // Cleanup self->status before returning.
    TF_SetStatus(self->status, TF_OK, "");
    return nullptr;
  } else {
    return PyArray_Return(reinterpret_cast<PyArrayObject*>(py_array));
  }
}

// Getter `device`.
static PyObject* EagerTensor_device(EagerTensor* self) {
  const char* device = TFE_TensorHandleDeviceName(self->handle, self->status);
  if (MaybeRaiseExceptionFromTFStatus(self->status, PyExc_ValueError)) {
    // Cleanup self->status before returning.
    TF_SetStatus(self->status, TF_OK, "");
    return nullptr;
  }
#if PY_MAJOR_VERSION >= 3
  return PyUnicode_FromString(device);
#else
  return PyBytes_FromString(device);
#endif
}

// Getter `backing_device`.
static PyObject* EagerTensor_backing_device(EagerTensor* self) {
  const char* device =
      TFE_TensorHandleBackingDeviceName(self->handle, self->status);
  if (MaybeRaiseExceptionFromTFStatus(self->status, PyExc_ValueError)) {
    // Cleanup self->status before returning.
    TF_SetStatus(self->status, TF_OK, "");
    return nullptr;
  }
#if PY_MAJOR_VERSION >= 3
  return PyUnicode_FromString(device);
#else
  return PyBytes_FromString(device);
#endif
}

static PyGetSetDef EagerTensor_getseters[] = {
    {const_cast<char*>("_id"), (getter)EagerTensor_getid, nullptr,
     const_cast<char*>("_id"), nullptr},
    {const_cast<char*>("device"), (getter)EagerTensor_device, nullptr,
     const_cast<char*>("device"), nullptr},
    {const_cast<char*>("backing_device"), (getter)EagerTensor_backing_device,
     nullptr, const_cast<char*>("backing_device"), nullptr},
    {const_cast<char*>("_handle_data"), (getter)EagerTensor_tensor_handle,
     (setter)EagerTensor_settensor_handle, const_cast<char*>("_tensor_handle"),
     nullptr},
    {const_cast<char*>("_tensor_shape"), (getter)EagerTensor_tensor_shape,
     (setter)EagerTensor_settensor_shape, const_cast<char*>("_tensor_shape"),
     nullptr},
    {nullptr} /* Sentinel */
};

#if PY_MAJOR_VERSION < 3
// Only used for Python2 since Python3 seems to set the __dict__ correctly.
static PyMemberDef EagerTensor_members[] = {
    {const_cast<char*>("__dict__"), T_OBJECT, offsetof(EagerTensor, dict),
     READONLY},
    {nullptr},
};
#endif

static PyMethodDef EagerTensor_methods[] = {
    {"_numpy", (PyCFunction)EagerTensor_numpy, METH_NOARGS,
     PyDoc_STR("_numpy")},
    {"_datatype_enum", (PyCFunction)EagerTensor_datatype_enum, METH_NOARGS,
     PyDoc_STR("_datatype_enum")},
    {"_shape_tuple", (PyCFunction)EagerTensor_shape_tuple, METH_NOARGS,
     PyDoc_STR("_shape_tuple")},
    {"_rank", (PyCFunction)EagerTensor_rank, METH_NOARGS, PyDoc_STR("_rank")},
    {"_copy_to_device", (PyCFunction)EagerTensor_copy_to_device,
     METH_VARARGS | METH_KEYWORDS, PyDoc_STR("_copy_to_device")},
    {"_num_elements", (PyCFunction)EagerTensor_num_elements, METH_NOARGS,
     PyDoc_STR("_num_elements")},
    {nullptr, nullptr},
};

static int EagerTensor_getbuffer(EagerTensor* self, Py_buffer* view,
                                 int flags) {
  if ((flags & PyBUF_WRITABLE) == PyBUF_WRITABLE) {
    PyErr_SetString(PyExc_BufferError, "EagerTensor is not writable.");
    return -1;
  }

  // TensorHandleToNumpy is zero-copy for everything but DT_RESOURCE and
  // DT_STRING so the following is only slightly slower than a NumPy-free
  // implementation.
  auto py_array = tensorflow::make_safe(
      TFE_TensorHandleToNumpy(self->handle, self->status));
  if (MaybeRaiseExceptionFromTFStatus(self->status, PyExc_BufferError)) {
    // Cleanup self->status before returning.
    TF_SetStatus(self->status, TF_OK, "");
    return -1;
  }
  if (PyObject_GetBuffer(py_array.get(), view, flags) < 0) {
    return -1;
  }
  view->readonly = 1;
  return 0;
}

static PyBufferProcs EagerTensor_as_buffer = {
#if PY_MAJOR_VERSION < 3
    nullptr, nullptr, nullptr, nullptr,
#endif
    (getbufferproc)EagerTensor_getbuffer,
    // Never called because getbufferproc delegates to NumPy.
    (releasebufferproc) nullptr};

// Note that here we are trying to dynamically create a new class as a subclass
// of a "HEAPTYPE" class that is itself created in python code and passed in at
// runtime. This is fairly atypical and undocumented.
//
// We use the following strategy for this. Unfortunately, we have to use
// different approaches for python2.x vs python3.x
// For python2.x, we create the class as a static type and set its tp_base to
// the passed in type. Unfortunately setting tp_flags to include
// Py_TPFLAGS_HEAPTYPE does not work by itself since it needs some more
// initialization of the underlying PyHeapTypeObject and not doing that leads to
// some random crashes especially during garbage collection.
// python3.x explicitly disables a static subclass of a HEAPTYPE base class.
// However it provides a new function, PyType_FromSpecWithBases, to create
// types dynamically.

// Type object for EagerTensor. This is set by TFE_Py_InitEagerTensor.
PyTypeObject* EagerTensorType = nullptr;

#if PY_MAJOR_VERSION >= 3
static PyType_Slot EagerTensor_Type_slots[] = {
    {Py_tp_dealloc, reinterpret_cast<void*>(EagerTensor_dealloc)},
    {Py_tp_methods, reinterpret_cast<void*>(EagerTensor_methods)},
    {Py_tp_getset, reinterpret_cast<void*>(EagerTensor_getseters)},
    {Py_tp_init, reinterpret_cast<void*>(EagerTensor_init)},
    {0, nullptr},
};
#else

#define EAGER_TENSOR_TPFLAGS (Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_NEWBUFFER)

// TODO(agarwal): support active_trace.
static PyTypeObject _EagerTensorType = {
    // clang-format off
    PyVarObject_HEAD_INIT(nullptr, 0)
    // clang-format on
    "EagerTensor",                      /* tp_name */
    sizeof(EagerTensor),                /* tp_basicsize */
    0,                                  /* tp_itemsize */
    (destructor)EagerTensor_dealloc,    /* tp_dealloc */
    nullptr,                            /* tp_print */
    nullptr,                            /* tp_getattr */
    nullptr,                            /* tp_setattr */
    nullptr,                            /* tp_compare */
    nullptr,                            /* tp_repr */
    nullptr,                            /* tp_as_number */
    nullptr,                            /* tp_as_sequence */
    nullptr,                            /* tp_as_mapping */
    nullptr,                            /* tp_hash */
    nullptr,                            /* tp_call */
    nullptr,                            /* tp_str */
    nullptr,                            /* tp_getattro */
    nullptr,                            /* tp_setattro */
    &EagerTensor_as_buffer,             /* tp_as_buffer */
    EAGER_TENSOR_TPFLAGS,               /* tp_flags */
    nullptr,                            /* tp_doc */
    nullptr,                            /* tp_traverse */
    nullptr,                            /* tp_clear */
    nullptr,                            /* tp_richcompare */
    offsetof(EagerTensor, weakreflist), /* tp_weaklistoffset */
    nullptr,                            /* tp_iter */
    nullptr,                            /* tp_iternext */
    EagerTensor_methods,                /* tp_methods */
    EagerTensor_members,                /* tp_members */
    EagerTensor_getseters,              /* tp_getset */
    nullptr,                            /* tp_base */
    nullptr,                            /* tp_dict */
    nullptr,                            /* tp_descr_get */
    nullptr,                            /* tp_descr_set */
    offsetof(EagerTensor, dict),        /* tp_dictoffset */
    (initproc)EagerTensor_init,         /* tp_init */
    nullptr,                            /* tp_alloc */
    nullptr,                            /* tp_new */
};

#endif

}  // extern "C"

bool EagerTensor_CheckExact(const PyObject* o) {
  return Py_TYPE(o) == EagerTensorType;
}

TFE_TensorHandle* EagerTensor_Handle(const PyObject* o) {
  return reinterpret_cast<const EagerTensor*>(o)->handle;
}

PyObject* EagerTensorFromHandle(TFE_TensorHandle* handle) {
  if (handle == nullptr) {
    return nullptr;
  }
  EagerTensor* t = reinterpret_cast<EagerTensor*>(
      EagerTensorType->tp_new(EagerTensorType, Py_None, Py_None));
  if (t != nullptr) {
    t->id = get_uid();
    Py_INCREF(Py_None);
    t->handle_data = Py_None;
    Py_INCREF(Py_None);
    t->tensor_shape = Py_None;
    t->handle = handle;
    t->status = TF_NewStatus();
    t->weakreflist = nullptr;
    PyObject* py_context = GetPyEagerContext();
    if (py_context == nullptr) {
      LOG(ERROR) << "Cannot create an eager tensor before eager context has "
                    "been set or after it has been deleted";
      return nullptr;
    }
    t->context = py_context;

    if (!MaybeInvokeCreatedOnEagerTensorProfiler(t)) {
      return nullptr;
    }
  }
  return reinterpret_cast<PyObject*>(t);
}

tensorflow::int64 PyEagerTensor_ID(const PyObject* tensor) {
  DCHECK(EagerTensor_CheckExact(tensor));
  return reinterpret_cast<const EagerTensor*>(tensor)->id;
}

tensorflow::DataType PyEagerTensor_Dtype(const PyObject* tensor) {
  DCHECK(EagerTensor_CheckExact(tensor));
  return static_cast<tensorflow::DataType>(TFE_TensorHandleDataType(
      reinterpret_cast<const EagerTensor*>(tensor)->handle));
}

tensorflow::int64 PyEagerTensor_NumElements(const PyObject* tensor) {
  DCHECK(EagerTensor_CheckExact(tensor));
  const EagerTensor* as_c_eager_tensor =
      reinterpret_cast<const EagerTensor*>(tensor);
  tensorflow::int64 result = TFE_TensorHandleNumElements(
      as_c_eager_tensor->handle, as_c_eager_tensor->status);

  if (MaybeRaiseExceptionFromTFStatus(as_c_eager_tensor->status,
                                      PyExc_ValueError)) {
    // Cleanup status before returning.
    TF_SetStatus(as_c_eager_tensor->status, TF_OK, "");
    return -1;
  }

  return result;
}

PyObject* TFE_Py_InitEagerTensor(PyObject* base_class) {
  if (!PyType_Check(base_class)) {
    PyErr_SetString(
        PyExc_TypeError,
        tensorflow::strings::StrCat(
            "Expecting a class definition for `base_class` passed to ",
            "TFE_InitEagerTensor. Got ", Py_TYPE(base_class)->tp_name)
            .c_str());
    return nullptr;
  }
  // Note that we allocated kMaxEagerTensorParentSize bytes of unused space in
  // EagerTensor to allow for the space usage of the base class.
  PyTypeObject* base_class_type = reinterpret_cast<PyTypeObject*>(base_class);
  if (base_class_type->tp_basicsize > kMaxEagerTensorParentSize) {
    PyErr_SetString(
        PyExc_TypeError,
        tensorflow::strings::StrCat(
            "Unable to create subclass EagerTensor from base class ",
            Py_TYPE(base_class)->tp_name,
            ". Need its size to be <= ", kMaxEagerTensorParentSize)
            .c_str());
    return nullptr;
  }
  if (base_class_type->tp_itemsize != 0) {
    PyErr_SetString(
        PyExc_TypeError,
        tensorflow::strings::StrCat(
            "Unable to create subclass EagerTensor from base class ",
            Py_TYPE(base_class)->tp_name,
            " which supports variable length instances.")
            .c_str());
    return nullptr;
  }
  Py_INCREF(base_class);
#if PY_MAJOR_VERSION >= 3
  PyObject* bases = PyTuple_New(1);
  PyTuple_SET_ITEM(bases, 0, base_class);

  tensorflow::Safe_PyObjectPtr base_class_module(
      PyObject_GetAttrString(base_class, "__module__"));
  const char* module = nullptr;
  if (PyErr_Occurred()) {
    PyErr_Clear();
    module = "__builtin__";
  } else {
    module = PyBytes_AsString(base_class_module.get());
    if (module == nullptr) {
      PyErr_Clear();
      module = PyUnicode_AsUTF8(base_class_module.get());
      if (module == nullptr) {
        PyErr_Clear();
        module = "__builtin__";
      }
    }
  }

  // NOTE: The c_str from this string needs to outlast the function, hence is
  // static.
  static tensorflow::string fully_qualified_name =
      tensorflow::strings::StrCat(module, ".EagerTensor");

  static PyType_Spec EagerTensor_Type_spec = {
      fully_qualified_name.c_str(), sizeof(EagerTensor), 0,
      Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HEAPTYPE, EagerTensor_Type_slots};

  EagerTensorType = reinterpret_cast<PyTypeObject*>(
      PyType_FromSpecWithBases(&EagerTensor_Type_spec, bases));
  if (PyErr_Occurred()) {
    return nullptr;
  }
  if (EagerTensorType == nullptr) {
    PyErr_SetString(PyExc_RuntimeError, "Error while creating EagerTensorType");
    return nullptr;
  }
  EagerTensorType->tp_dictoffset = offsetof(EagerTensor, dict);
  EagerTensorType->tp_as_buffer = &EagerTensor_as_buffer;
#else
  _EagerTensorType.tp_base = base_class_type;

  if (PyType_Ready(&_EagerTensorType) < 0) {
    if (PyErr_Occurred()) return nullptr;
    PyErr_SetString(PyExc_RuntimeError,
                    "Error while creating EagerTensor type.");
    return nullptr;
  }
  EagerTensorType = &_EagerTensorType;
  Py_INCREF(EagerTensorType);
#endif
  return reinterpret_cast<PyObject*>(EagerTensorType);
}

PyObject* TFE_Py_SetEagerTensorProfiler(PyObject* profiler) {
  Py_XDECREF(eager_tensor_profiler);

  if (profiler == Py_None) {
    eager_tensor_profiler = nullptr;
  } else {
    eager_tensor_profiler = profiler;
    Py_INCREF(eager_tensor_profiler);
  }
  Py_RETURN_NONE;
}

PyObject* TFE_Py_TensorShapeSlice(PyObject* tensors, int slice_dim) {
  if (!PyList_Check(tensors) && !PyTuple_Check(tensors)) {
    PyErr_SetString(PyExc_TypeError,
                    tensorflow::strings::StrCat(
                        "tensors argument must be a list or a tuple. Got \"",
                        Py_TYPE(tensors)->tp_name, "\"")
                        .c_str());
    return nullptr;
  }
  if (slice_dim < 0) {
    PyErr_SetString(
        PyExc_ValueError,
        tensorflow::strings::StrCat("Slice dimension must be non-negative. "
                                    "Got ",
                                    slice_dim)
            .c_str());
    return nullptr;
  }

  Py_ssize_t num_tensors = PySequence_Fast_GET_SIZE(tensors);
  int64_t num_tensors_int = static_cast<int64_t>(num_tensors);
  auto tensor = tensorflow::make_safe(TF_AllocateTensor(
      TF_INT32, &num_tensors_int, /*num_dims=*/1, /*len=*/4 * num_tensors_int));
  int32_t* data = reinterpret_cast<int32_t*>(TF_TensorData(tensor.get()));
  auto status = tensorflow::make_safe(TF_NewStatus());
  for (Py_ssize_t i = 0; i < num_tensors; ++i) {
    PyObject* tensor_obj = PySequence_Fast_GET_ITEM(tensors, i);
    if (!EagerTensor_CheckExact(tensor_obj)) {
      PyErr_SetString(PyExc_TypeError,
                      tensorflow::strings::StrCat(
                          "Expected a list of EagerTensors but "
                          "element ",
                          i, " has type \"", Py_TYPE(tensor_obj)->tp_name, "\"")
                          .c_str());
      return nullptr;
    }

    EagerTensor* t = reinterpret_cast<EagerTensor*>(tensor_obj);
    TFE_TensorHandle* handle = t->handle;
    int num_dims = TFE_TensorHandleNumDims(handle, status.get());
    if (MaybeRaiseExceptionFromTFStatus(status.get(), PyExc_ValueError)) {
      return nullptr;
    }
    if (slice_dim >= num_dims) {
      PyErr_SetString(
          PyExc_IndexError,
          tensorflow::strings::StrCat("Slice dimension (", slice_dim,
                                      ") must be smaller than rank of all "
                                      "tensors, but tensor at index ",
                                      i, " has rank ", num_dims)
              .c_str());
      return nullptr;
    }
    int64_t dim = TFE_TensorHandleDim(handle, slice_dim, status.get());
    if (MaybeRaiseExceptionFromTFStatus(status.get(), PyExc_ValueError)) {
      return nullptr;
    }
    data[i] = dim;
  }

  TFE_TensorHandle* handle = TFE_NewTensorHandle(tensor.get(), status.get());
  if (TF_GetCode(status.get()) != TF_OK) {
    PyErr_SetString(
        PyExc_RuntimeError,
        tensorflow::strings::StrCat("Failed to construct new tensor handle: ",
                                    TF_Message(status.get()))
            .c_str());
    return nullptr;
  }

  return EagerTensorFromHandle(handle);
}

PyObject* TFE_Py_TensorShapeOnDevice(PyObject* tensor) {
  if (!EagerTensor_CheckExact(tensor)) {
    PyErr_SetString(
        PyExc_TypeError,
        tensorflow::strings::StrCat("Expected an EagerTensors but got type \"",
                                    Py_TYPE(tensor)->tp_name, "\"")
            .c_str());
    return nullptr;
  }
  TFE_TensorHandle* handle = EagerTensor_Handle(tensor);

  auto status = tensorflow::make_safe(TF_NewStatus());
  TFE_TensorDebugInfo* debug_info =
      TFE_TensorHandleTensorDebugInfo(handle, status.get());
  if (TF_GetCode(status.get()) != TF_OK) {
    PyErr_SetString(
        PyExc_RuntimeError,
        tensorflow::strings::StrCat("Error retrieving tensor's device shape: ",
                                    TF_Message(status.get()))
            .c_str());
    return nullptr;
  }

  int rank = TFE_TensorDebugInfoOnDeviceNumDims(debug_info);
  PyObject* shape = PyTuple_New(rank);
  for (int i = 0; i < rank; ++i) {
    tensorflow::int64 dim_size = TFE_TensorDebugInfoOnDeviceDim(debug_info, i);
    PyTuple_SET_ITEM(shape, i, PyLong_FromLongLong(dim_size));
  }
  TFE_DeleteTensorDebugInfo(debug_info);

  return shape;
}
