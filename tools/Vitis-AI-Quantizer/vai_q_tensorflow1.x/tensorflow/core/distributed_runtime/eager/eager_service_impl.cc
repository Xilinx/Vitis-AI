/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/distributed_runtime/eager/eager_service_impl.h"

#include "absl/memory/memory.h"
#include "tensorflow/c/c_api_internal.h"
#include "tensorflow/c/tf_status_helper.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/common_runtime/eager/context.h"
#include "tensorflow/core/common_runtime/eager/eager_operation.h"
#include "tensorflow/core/common_runtime/eager/execute.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/common_runtime/process_util.h"
#include "tensorflow/core/distributed_runtime/eager/remote_mgr.h"
#include "tensorflow/core/distributed_runtime/eager/remote_tensor_handle.h"
#include "tensorflow/core/distributed_runtime/rpc/rpc_rendezvous_mgr.h"
#include "tensorflow/core/distributed_runtime/server_lib.h"
#include "tensorflow/core/distributed_runtime/session_mgr.h"
#include "tensorflow/core/distributed_runtime/worker_cache.h"
#include "tensorflow/core/distributed_runtime/worker_cache_wrapper.h"
#include "tensorflow/core/distributed_runtime/worker_env.h"
#include "tensorflow/core/framework/rendezvous.h"
#include "tensorflow/core/lib/core/error_codes.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/cleanup.h"
#include "tensorflow/core/lib/random/random.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/cpu_info.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/host_info.h"
#include "tensorflow/core/profiler/lib/traceme.h"

namespace tensorflow {
namespace eager {

namespace {
Status GetNumRetvals(tensorflow::EagerContext* context, const string& op_name,
                     const google::protobuf::Map<string, tensorflow::AttrValue>& attrs,
                     int* num_retvals) {
  const tensorflow::OpRegistrationData* op_reg_data = nullptr;
  auto status = tensorflow::OpRegistry::Global()->LookUp(op_name, &op_reg_data);
  if (errors::IsNotFound(status)) {
    status = context->FindFunctionOpData(op_name, &op_reg_data);
  }
  TF_RETURN_IF_ERROR(status);

  const tensorflow::OpDef& op_def = op_reg_data->op_def;

  for (const auto& output_arg : op_def.output_arg()) {
    if (!output_arg.number_attr().empty()) {
      auto iter = attrs.find(output_arg.number_attr());
      if (iter == attrs.end()) {
        return errors::InvalidArgument("Unable to find number_attr ",
                                       output_arg.number_attr(),
                                       " for Op: ", op_name);
      }
      *num_retvals += iter->second.i();
    } else if (!output_arg.type_list_attr().empty()) {
      auto iter = attrs.find(output_arg.type_list_attr());
      if (iter == attrs.end()) {
        return errors::InvalidArgument("Unable to find type_list_attr ",
                                       output_arg.type_list_attr(),
                                       " for Op: ", op_name);
      }
      *num_retvals += iter->second.list().type_size();
    } else {
      *num_retvals += 1;
    }
  }

  return Status::OK();
}
}  // namespace

Status EagerServiceImpl::CreateContext(const CreateContextRequest* request,
                                       CreateContextResponse* response) {
  // make sure env_ , env_->rendezvous_mgr available
  if (env_ == nullptr || env_->rendezvous_mgr == nullptr) {
    return tensorflow::errors::Internal(
        "invalid eager env_ or env_->rendezvous_mgr.");
  }
  std::vector<DeviceAttributes> cluster_device_attributes;
  cluster_device_attributes.reserve(
      request->cluster_device_attributes().size());
  for (const auto& cluster_device : request->cluster_device_attributes()) {
    cluster_device_attributes.push_back(cluster_device);
  }

  auto* r = env_->rendezvous_mgr->Find(request->context_id());
  auto session_name =
      tensorflow::strings::StrCat("eager_", request->context_id());
  TF_RETURN_IF_ERROR(env_->session_mgr->CreateSession(
      session_name, request->server_def(), request->cluster_device_attributes(),
      true));

  std::shared_ptr<WorkerSession> worker_session;
  TF_RETURN_IF_ERROR(env_->session_mgr->WorkerSessionForSession(
      session_name, &worker_session));

  tensorflow::DeviceMgr* device_mgr = worker_session->device_mgr();

  // Initialize remote tensor communication based on worker session.
  TF_RETURN_IF_ERROR(r->Initialize(worker_session.get()));

  std::function<Rendezvous*(const int64)> rendezvous_creator =
      [worker_session, this](const int64 step_id) {
        auto* r = env_->rendezvous_mgr->Find(step_id);
        r->Initialize(worker_session.get()).IgnoreError();
        return r;
      };

  LOG(INFO) << "Creating " << (request->async() ? "async" : "sync")
            << " eager service context with rendezvous_id on host "
            << port::Hostname();
  tensorflow::EagerContext* ctx = new tensorflow::EagerContext(
      SessionOptions(),
      tensorflow::ContextDevicePlacementPolicy::DEVICE_PLACEMENT_SILENT,
      tensorflow::ContextMirroringPolicy::MIRRORING_NONE, request->async(),
      device_mgr, false, r, GetDefaultCustomKernelCreator(),
      worker_session->cluster_flr.get());
  // Ownership will be transferred to the ServerContext, or else in an error
  // case ctx will be deleted by this unref.
  core::ScopedUnref unref_ctx(ctx);

  std::vector<string> remote_workers;
  worker_session->worker_cache->ListWorkers(&remote_workers);
  remote_workers.erase(std::remove(remote_workers.begin(), remote_workers.end(),
                                   worker_session->worker_name),
                       remote_workers.end());

  std::unique_ptr<tensorflow::eager::EagerClientCache> remote_eager_workers;
  TF_RETURN_IF_ERROR(
      worker_session->worker_cache->GetEagerClientCache(&remote_eager_workers));

  auto remote_mgr =
      absl::make_unique<tensorflow::eager::RemoteMgr>(/*is_master=*/false, ctx);
  Status s = ctx->InitializeRemoteWorker(
      std::move(remote_eager_workers), worker_session->remote_device_mgr(),
      remote_workers, request->context_id(), std::move(rendezvous_creator),
      std::move(remote_mgr));
  if (!s.ok()) {
    VLOG(1) << "EagerContext::InitializeRemoteWorker failed with "
            << s.ToString();
    return s;
  }

  std::vector<DeviceAttributes> device_attributes;
  device_mgr->ListDeviceAttributes(&device_attributes);

  for (const auto& da : device_attributes) {
    *response->add_device_attributes() = da;
  }
  {
    mutex_lock l(contexts_mu_);
    if (contexts_.find(request->context_id()) != contexts_.end()) {
      return errors::InvalidArgument("EagerService:CreateContext failed. ",
                                     "Context id: <", request->context_id(),
                                     "> already exists.");
    }
    contexts_.emplace(request->context_id(),
                      new ServerContext(ctx, request->keep_alive_secs(), env_));
  }

  return Status::OK();
}

Status EagerServiceImpl::CreateMasterContext(
    const tensorflow::uint64 context_id, EagerContext* context) {
  {
    mutex_lock l(contexts_mu_);
    auto iter = contexts_.find(context_id);
    if (iter != contexts_.end()) {
      return errors::InvalidArgument(
          "EagerService:CreateMasterContext failed. ", "Context id: <",
          context_id, "> already exists.");
    }
  }
  ServerContext* server_context =
      ServerContext::CreateMasterContext(context, env_);
  mutex_lock l(contexts_mu_);
  contexts_.emplace(context_id, server_context);
  return Status::OK();
}

Status TensorHandleShape(TensorHandle* handle, TensorShapeProto* proto) {
  const tensorflow::Tensor* t = nullptr;

  // TODO(nareshmodi): This call makes async calls sync calls. Fix this.
  TF_RETURN_IF_ERROR(handle->Tensor(&t));

  t->shape().AsProto(proto);

  return Status::OK();
}

Status EagerServiceImpl::ExecuteOp(const Operation& operation,
                                   EagerContext* eager_context,
                                   QueueResponse* queue_response) {
  std::unique_ptr<tensorflow::EagerOperation> op;
  const char* name = operation.name().c_str();  // Shorthand
  const tensorflow::AttrTypeMap* types;
  bool is_function = false;
  TF_RETURN_IF_ERROR(tensorflow::AttrTypeMapForOp(name, &types, &is_function));
  if (is_function && !eager_context->FindFunctionByName(name)) {
    return errors::NotFound(
        "'", name,
        "' is neither a type of a primitive operation nor a name "
        "of a function registered in binary running on ",
        port::Hostname(),
        ". Make sure the operation or function is "
        "registered in the binary running in this process.");
  }
  op.reset(
      new tensorflow::EagerOperation(eager_context, name, is_function, types));

  TF_RETURN_IF_ERROR(op->SetDeviceName(operation.device().c_str()));

  {
    profiler::TraceMe activity("EagerService:RemoteTensorHandleInternal",
                               profiler::TraceMeLevel::kVerbose);
    for (const auto& remote_handle : operation.inputs()) {
      tensorflow::TensorHandle* handle;
      TF_RETURN_IF_ERROR(
          eager_context->RemoteMgr()->DeserializeRemoteTensorHandle(
              remote_handle, &handle));
      op->AddInput(handle);
      // Unref handle since it has a ref as an input now.
      handle->Unref();
    }
  }

  for (const auto& attr : operation.attrs()) {
    op->MutableAttrs()->Set(attr.first, attr.second);
  }

  int num_retvals = 0;
  // TODO(nareshmodi): Consider caching this.
  TF_RETURN_IF_ERROR(GetNumRetvals(eager_context, operation.name(),
                                   operation.attrs(), &num_retvals));

  tensorflow::gtl::InlinedVector<tensorflow::TensorHandle*, 2> retvals(
      num_retvals);
  VLOG(3) << "ServerContext: Calling EagerExecute for op " << operation.id();
  TF_RETURN_IF_ERROR(EagerExecute(op.get(), &retvals, &num_retvals));
  retvals.resize(num_retvals);

  eager_context->RemoteMgr()->AddOperationOutputs(retvals, operation.id());

  for (auto* handle : retvals) {
    TF_RETURN_IF_ERROR(TensorHandleShape(handle, queue_response->add_shape()));
  }

  return Status::OK();
}

Status EagerServiceImpl::Enqueue(const EnqueueRequest* request,
                                 EnqueueResponse* response) {
  profiler::TraceMe activity(
      [&] {
        return absl::StrCat("EagerService:Enqueue:", request->DebugString());
      },
      profiler::TraceMeLevel::kInfo);
  ServerContext* context = nullptr;
  TF_RETURN_IF_ERROR(GetServerContext(request->context_id(), &context));
  core::ScopedUnref context_unref(context);

  auto executor = context->Context()->Executor();
  for (const auto& item : request->queue()) {
    auto* queue_response = response->add_queue_response();
    if (item.has_operation()) {
      TF_RETURN_IF_ERROR(
          ExecuteOp(item.operation(), context->Context(), queue_response));
    } else if (item.has_handle_to_decref()) {
      auto handle_to_decref = absl::make_unique<RemoteTensorHandleInternal>(
          item.handle_to_decref());
      auto node = absl::make_unique<ClientTensorHandleDeleteNode>(
          context, std::move(handle_to_decref));
      TF_RETURN_IF_ERROR(
          executor->Async()
              ? context->Context()->Executor()->Add(std::move(node))
              : node->Run());
    } else {
      TF_RETURN_IF_ERROR(SendTensor(item.send_tensor(), context->Context()));
    }
  }

  return Status::OK();
}

Status EagerServiceImpl::WaitQueueDone(const WaitQueueDoneRequest* request,
                                       WaitQueueDoneResponse* response) {
  ServerContext* context = nullptr;
  TF_RETURN_IF_ERROR(GetServerContext(request->context_id(), &context));
  core::ScopedUnref context_unref(context);

  if (request->op_id_size() > 0) {
    return errors::Unimplemented(
        "EagerServiceImpl::WaitQueueDone is not "
        "implemented for particular op IDs.");
  }
  return context->Context()->Executor()->WaitForAllPendingNodes();
}

Status EagerServiceImpl::KeepAlive(const KeepAliveRequest* request,
                                   KeepAliveResponse* response) {
  ServerContext* context = nullptr;
  TF_RETURN_IF_ERROR(GetServerContext(request->context_id(), &context));
  core::ScopedUnref context_unref(context);

  return Status::OK();
}

Status EagerServiceImpl::CloseContext(const CloseContextRequest* request,
                                      CloseContextResponse* response) {
  VLOG(1) << "Executing EagerService::CloseContext for context "
          << request->context_id();
  ServerContext* context = nullptr;
  if (!GetServerContext(request->context_id(), &context).ok()) {
    // Swallow the error here.
    return Status::OK();
  }

  core::ScopedUnref context_unref(context);

  mutex_lock l(contexts_mu_);
  contexts_.erase(request->context_id());

  // GetServerContext returns a newly Reffed copy of ServerContext, which is
  // unreffed by context_unref. Additionally, we need to unref it one time since
  // we are releasing it from the map.
  context->Unref();

  return Status::OK();
}

Status EagerServiceImpl::RegisterFunction(
    const RegisterFunctionRequest* request,
    RegisterFunctionResponse* response) {
  ServerContext* context = nullptr;
  TF_RETURN_IF_ERROR(GetServerContext(request->context_id(), &context));
  core::ScopedUnref context_unref(context);

  return context->Context()->AddFunctionDef(request->function_def());
}

Status EagerServiceImpl::SendTensor(const SendTensorRequest* request,
                                    SendTensorResponse* response) {
  ServerContext* context = nullptr;
  TF_RETURN_IF_ERROR(GetServerContext(request->context_id(), &context));
  core::ScopedUnref context_unref(context);

  tensorflow::gtl::InlinedVector<tensorflow::TensorHandle*, 2> tensors;
  for (const auto& tensor_proto : request->tensors()) {
    Tensor tensor;
    if (!tensor.FromProto(tensor_proto)) {
      return errors::InvalidArgument("Unable to parse tensor proto");
    }

    TensorHandle* tensor_handle = nullptr;
    TF_RETURN_IF_ERROR(TensorHandle::CreateLocalHandle(tensor, &tensor_handle));
    TensorHandle* copied_handle = nullptr;
    EagerContext* ctx = context->Context();
    Device* device;
    TF_RETURN_IF_ERROR(
        ctx->FindDeviceFromName(request->device_name().c_str(), &device));
    TF_RETURN_IF_ERROR(EagerCopyToDevice(tensor_handle, ctx, ctx->Executor(),
                                         device, false, &copied_handle));
    tensors.push_back(copied_handle);
    tensor_handle->Unref();
  }

  context->Context()->RemoteMgr()->AddOperationOutputs(tensors,
                                                       request->op_id());

  return Status::OK();
}

Status EagerServiceImpl::SendTensor(const SendTensorOp& send_tensor,
                                    EagerContext* eager_context) {
  tensorflow::gtl::InlinedVector<tensorflow::TensorHandle*, 2> tensors;
  for (const auto& tensor_proto : send_tensor.tensors()) {
    Tensor tensor;
    if (!tensor.FromProto(tensor_proto)) {
      return errors::InvalidArgument("Unable to parse tensor proto");
    }

    TensorHandle* tensor_handle = nullptr;
    TF_RETURN_IF_ERROR(TensorHandle::CreateLocalHandle(tensor, &tensor_handle));
    TensorHandle* copied_handle = nullptr;
    Device* device;
    TF_RETURN_IF_ERROR(eager_context->FindDeviceFromName(
        send_tensor.device_name().c_str(), &device));
    TF_RETURN_IF_ERROR(EagerCopyToDevice(tensor_handle, eager_context,
                                         eager_context->Executor(), device,
                                         false, &copied_handle));
    tensors.push_back(copied_handle);
    tensor_handle->Unref();
  }

  eager_context->RemoteMgr()->AddOperationOutputs(tensors, send_tensor.op_id());

  return Status::OK();
}

tensorflow::Status EagerServiceImpl::GetServerContext(
    uint64 context_id, ServerContext** server_context) {
  mutex_lock l(contexts_mu_);
  auto iter = contexts_.find(context_id);
  if (iter == contexts_.end()) {
    *server_context = nullptr;
    return errors::InvalidArgument(strings::Printf(
        "Unable to find a context_id matching the specified one "
        "(%lld). Perhaps the worker was restarted, or the context was GC'd?",
        context_id));
  }

  *server_context = iter->second;
  (*server_context)->Ref();

  (*server_context)->RecordAccess();

  return Status::OK();
}

}  // namespace eager
}  // namespace tensorflow
