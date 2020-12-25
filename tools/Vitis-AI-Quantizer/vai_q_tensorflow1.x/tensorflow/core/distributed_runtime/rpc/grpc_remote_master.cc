/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/distributed_runtime/rpc/grpc_remote_master.h"

#include <utility>

#include "tensorflow/core/distributed_runtime/call_options.h"
#include "tensorflow/core/distributed_runtime/master_interface.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_master_service_impl.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/tracing.h"
#include "tensorflow/core/profiler/lib/traceme.h"
#include "tensorflow/core/protobuf/master.pb.h"

namespace tensorflow {

// GrpcRemoteMaster is an implementation of the MasterInterface
// that uses gRPC to talk to the Master service.
class GrpcRemoteMaster : public MasterInterface {
  using MasterServiceStub = grpc::MasterService::Stub;

 public:
  explicit GrpcRemoteMaster(const SharedGrpcChannelPtr& client_channel)
      : stub_(grpc::MasterService::NewStub(client_channel)) {}

  ~GrpcRemoteMaster() override {}

  Status CreateSession(CallOptions* call_options,
                       const CreateSessionRequest* request,
                       CreateSessionResponse* response) override {
    return CallWithRetry(call_options, request, response,
                         &MasterServiceStub::CreateSession);
  }

  Status ExtendSession(CallOptions* call_options,
                       const ExtendSessionRequest* request,
                       ExtendSessionResponse* response) override {
    return CallWithRetry(call_options, request, response,
                         &MasterServiceStub::ExtendSession);
  }

  Status PartialRunSetup(CallOptions* call_options,
                         const PartialRunSetupRequest* request,
                         PartialRunSetupResponse* response) override {
    return CallWithRetry(call_options, request, response,
                         &MasterServiceStub::PartialRunSetup);
  }

  Status RunStep(CallOptions* call_options, RunStepRequestWrapper* request,
                 MutableRunStepResponseWrapper* response) override {
    return CallWithRetry(call_options, &request->ToProto(),
                         get_proto_from_wrapper(response),
                         &MasterServiceStub::RunStep, "RunStep/Client");
  }

  Status CloseSession(CallOptions* call_options,
                      const CloseSessionRequest* request,
                      CloseSessionResponse* response) override {
    return CallWithRetry(call_options, request, response,
                         &MasterServiceStub::CloseSession);
  }

  Status ListDevices(CallOptions* call_options,
                     const ListDevicesRequest* request,
                     ListDevicesResponse* response) override {
    return CallWithRetry(call_options, request, response,
                         &MasterServiceStub::ListDevices);
  }

  Status Reset(CallOptions* call_options, const ResetRequest* request,
               ResetResponse* response) override {
    return CallWithRetry(call_options, request, response,
                         &MasterServiceStub::Reset);
  }

  Status MakeCallable(CallOptions* call_options,
                      const MakeCallableRequest* request,
                      MakeCallableResponse* response) override {
    return CallWithRetry(call_options, request, response,
                         &MasterServiceStub::MakeCallable);
  }
  Status RunCallable(CallOptions* call_options,
                     const RunCallableRequest* request,
                     RunCallableResponse* response) override {
    return CallWithRetry(call_options, request, response,
                         &MasterServiceStub::RunCallable);
  }
  Status ReleaseCallable(CallOptions* call_options,
                         const ReleaseCallableRequest* request,
                         ReleaseCallableResponse* response) override {
    return CallWithRetry(call_options, request, response,
                         &MasterServiceStub::ReleaseCallable);
  }

 private:
  // Start tracing, attaching a unique ID to both the trace and the RPC.
  profiler::TraceMe* NewTraceRpc(StringPiece name, ::grpc::ClientContext* ctx) {
    string trace_id = strings::StrCat(tracing::GetUniqueArg());
    ctx->AddMetadata(GrpcIdKey(), trace_id);
    return new profiler::TraceMe(
        [&] { return strings::StrCat(name, ":", trace_id); },
        profiler::TraceMeLevel::kInfo);
  }

  template <typename Request, typename Response>
  Status CallWithRetry(CallOptions* call_options, const Request* request,
                       Response* response,
                       ::grpc::Status (MasterServiceStub::*pfunc)(
                           ::grpc::ClientContext*, const Request&, Response*),
                       string trace_string = {}) {
    int64 timeout_in_ms = call_options->GetTimeout();
    int64 expired_time_micros = Env::Default()->NowMicros();
    if (timeout_in_ms > 0) {
      expired_time_micros += (timeout_in_ms / 1000.);
    }
    Status s;
    for (int num_retries = 0;; ++num_retries) {
      ::grpc::ClientContext ctx;
      std::unique_ptr<profiler::TraceMe> trace;
      if (!trace_string.empty()) {
        trace.reset(NewTraceRpc(trace_string, &ctx));
      }
      ctx.set_fail_fast(false);
      if (timeout_in_ms > 0) {
        // We do not modify the timeout here to match legacy behavior. However,
        // this could violate the contract of tensorflow::Session. If we retry
        // an RPC just before the deadline is exceeded, we will still set the
        // timeout to the original value. This leads to the overall timeout
        // being double what was expected.
        // TODO(b/117162170): investigate fixing this behavior for legacy and
        // gRPC RPC layers.
        ctx.set_deadline(gpr_time_from_millis(timeout_in_ms, GPR_TIMESPAN));
      }
      s = FromGrpcStatus((stub_.get()->*pfunc)(&ctx, *request, response));
      if (!errors::IsUnavailable(s)) {
        return s;
      }
      // TODO(b/117162170): we may want to make this configurable.
      constexpr int kMaxRetries = 10;
      LOG(WARNING) << "RPC failed with status = \"" << s
                   << "\" and grpc_error_string = \""
                   << ctx.debug_error_string() << "\", maybe retrying the RPC";
      if (num_retries >= kMaxRetries) {
        LOG(WARNING) << "Too many retries, returning last status: " << s;
        return s;
      }
      const int64 now_micros = Env::Default()->NowMicros();
      const int64 deadline_with_backoff_micros =
          now_micros + ComputeBackoffMicroseconds(num_retries);
      // Wait for a short period of time before retrying the RPC.  If our
      // backoff would put us past the RPC deadline, we truncate it to ensure
      // our RPC starts before the deadline.
      const auto backoff_until =
          (timeout_in_ms <= 0 ||
           expired_time_micros > deadline_with_backoff_micros)
              ? deadline_with_backoff_micros
              : expired_time_micros;
      Env::Default()->SleepForMicroseconds(backoff_until - now_micros);
      if (Env::Default()->NowMicros() > expired_time_micros &&
          timeout_in_ms > 0) {
        // If timeout_in_ms is set, exit the retry loop on timeout.
        return errors::DeadlineExceeded(ctx.debug_error_string());
      }
    }
  }

  std::unique_ptr<MasterServiceStub> stub_;
};

MasterInterface* NewGrpcMaster(const SharedGrpcChannelPtr& channel) {
  return new GrpcRemoteMaster(channel);
}

}  // namespace tensorflow
