/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/framework/resource_handle.h"
#include "tensorflow/core/framework/resource_handle.pb.h"
#include "tensorflow/core/lib/strings/strcat.h"

namespace tensorflow {

ResourceHandle::ResourceHandle() {}

ResourceHandle::ResourceHandle(const ResourceHandleProto& proto) {
  FromProto(proto);
}

ResourceHandle::~ResourceHandle() {}

void ResourceHandle::AsProto(ResourceHandleProto* proto) const {
  proto->set_device(device());
  proto->set_container(container());
  proto->set_name(name());
  proto->set_hash_code(hash_code());
  proto->set_maybe_type_name(maybe_type_name());
  for (const auto& dtype_and_shape_pair : dtypes_and_shapes_) {
    auto dtype_and_shape = proto->add_dtypes_and_shapes();
    dtype_and_shape->set_dtype(dtype_and_shape_pair.dtype);
    dtype_and_shape_pair.shape.AsProto(dtype_and_shape->mutable_shape());
  }
}

void ResourceHandle::FromProto(const ResourceHandleProto& proto) {
  set_device(proto.device());
  set_container(proto.container());
  set_name(proto.name());
  set_hash_code(proto.hash_code());
  set_maybe_type_name(proto.maybe_type_name());
  std::vector<DtypeAndPartialTensorShape> dtypes_and_shapes;
  for (const auto& dtype_and_shape : proto.dtypes_and_shapes()) {
    DataType dtype = dtype_and_shape.dtype();
    PartialTensorShape shape(dtype_and_shape.shape());
    dtypes_and_shapes.push_back(DtypeAndPartialTensorShape{dtype, shape});
  }
  dtypes_and_shapes_ = std::move(dtypes_and_shapes);
}

string ResourceHandle::SerializeAsString() const {
  ResourceHandleProto proto;
  AsProto(&proto);
  return proto.SerializeAsString();
}

bool ResourceHandle::ParseFromString(const string& s) {
  ResourceHandleProto proto;
  const bool status = proto.ParseFromString(s);
  if (status) FromProto(proto);
  return status;
}

string ResourceHandle::DebugString() const {
  return strings::StrCat("device: ", device(), " container: ", container(),
                         " name: ", name(), " hash_code: ", hash_code(),
                         " maybe_type_name: ", maybe_type_name());
}

string ProtoDebugString(const ResourceHandle& handle) {
  return handle.DebugString();
}

void EncodeResourceHandleList(const ResourceHandle* p, int64 n,
                              std::unique_ptr<port::StringListEncoder> e) {
  ResourceHandleProto proto;
  for (int i = 0; i < n; ++i) {
    p[i].AsProto(&proto);
    e->Append(proto);
  }
  e->Finalize();
}

bool DecodeResourceHandleList(std::unique_ptr<port::StringListDecoder> d,
                              ResourceHandle* ps, int64 n) {
  std::vector<uint32> sizes(n);
  if (!d->ReadSizes(&sizes)) return false;

  ResourceHandleProto proto;
  for (int i = 0; i < n; ++i) {
    if (!proto.ParseFromArray(d->Data(sizes[i]), sizes[i])) {
      return false;
    }
    ps[i].FromProto(proto);
  }
  return true;
}

}  // namespace tensorflow
