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

#include "tensorflow/core/lib/core/status.h"

#include "absl/strings/match.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"

namespace tensorflow {

TEST(Status, OK) {
  EXPECT_EQ(Status::OK().code(), error::OK);
  EXPECT_EQ(Status::OK().error_message(), "");
  TF_EXPECT_OK(Status::OK());
  TF_ASSERT_OK(Status::OK());
  EXPECT_EQ(Status::OK(), Status());
  Status s;
  EXPECT_TRUE(s.ok());
}

TEST(DeathStatus, CheckOK) {
  Status status(errors::InvalidArgument("Invalid"));
  ASSERT_DEATH(TF_CHECK_OK(status), "Invalid");
}

TEST(Status, Set) {
  Status status;
  status = Status(error::CANCELLED, "Error message");
  EXPECT_EQ(status.code(), error::CANCELLED);
  EXPECT_EQ(status.error_message(), "Error message");
}

TEST(Status, Copy) {
  Status a(errors::InvalidArgument("Invalid"));
  Status b(a);
  ASSERT_EQ(a.ToString(), b.ToString());
}

TEST(Status, Assign) {
  Status a(errors::InvalidArgument("Invalid"));
  Status b;
  b = a;
  ASSERT_EQ(a.ToString(), b.ToString());
}

TEST(Status, Update) {
  Status s;
  s.Update(Status::OK());
  ASSERT_TRUE(s.ok());
  Status a(errors::InvalidArgument("Invalid"));
  s.Update(a);
  ASSERT_EQ(s.ToString(), a.ToString());
  Status b(errors::Internal("Internal"));
  s.Update(b);
  ASSERT_EQ(s.ToString(), a.ToString());
  s.Update(Status::OK());
  ASSERT_EQ(s.ToString(), a.ToString());
  ASSERT_FALSE(s.ok());
}

TEST(Status, EqualsOK) { ASSERT_EQ(Status::OK(), Status()); }

TEST(Status, EqualsSame) {
  Status a(errors::InvalidArgument("Invalid"));
  Status b(errors::InvalidArgument("Invalid"));
  ASSERT_EQ(a, b);
}

TEST(Status, EqualsCopy) {
  const Status a(errors::InvalidArgument("Invalid"));
  const Status b = a;
  ASSERT_EQ(a, b);
}

TEST(Status, EqualsDifferentCode) {
  const Status a(errors::InvalidArgument("message"));
  const Status b(errors::Internal("message"));
  ASSERT_NE(a, b);
}

TEST(Status, EqualsDifferentMessage) {
  const Status a(errors::InvalidArgument("message"));
  const Status b(errors::InvalidArgument("another"));
  ASSERT_NE(a, b);
}

TEST(StatusGroup, OKStatusGroup) {
  StatusGroup c;
  c.Update(Status::OK());
  c.Update(Status::OK());
  ASSERT_EQ(c.as_summary_status(), Status::OK());
  ASSERT_EQ(c.as_concatenated_status(), Status::OK());
}

TEST(StatusGroup, AggregateWithSingleErrorStatus) {
  StatusGroup c;
  const Status internal(errors::Internal("Original error."));

  c.Update(internal);
  ASSERT_EQ(c.as_summary_status(), internal);

  Status concat_status = c.as_concatenated_status();
  ASSERT_EQ(concat_status.code(), internal.code());
  ASSERT_TRUE(absl::StrContains(concat_status.error_message(),
                                internal.error_message()));

  // Add derived error status
  const Status derived =
      StatusGroup::MakeDerived(errors::Internal("Derived error."));
  c.Update(derived);

  ASSERT_EQ(c.as_summary_status(), internal);

  concat_status = c.as_concatenated_status();
  ASSERT_EQ(concat_status.code(), internal.code());
  ASSERT_TRUE(absl::StrContains(concat_status.error_message(),
                                internal.error_message()));
}

TEST(StatusGroup, AggregateWithMultipleErrorStatus) {
  StatusGroup c;
  const Status internal(errors::Internal("Original error."));
  const Status cancelled(errors::Cancelled("Cancelled after 10 steps."));
  const Status aborted(errors::Aborted("Aborted after 10 steps."));

  c.Update(internal);
  c.Update(cancelled);
  c.Update(aborted);

  Status summary = c.as_summary_status();

  ASSERT_EQ(summary.code(), internal.code());
  ASSERT_TRUE(
      absl::StrContains(summary.error_message(), internal.error_message()));
  ASSERT_TRUE(
      absl::StrContains(summary.error_message(), cancelled.error_message()));
  ASSERT_TRUE(
      absl::StrContains(summary.error_message(), aborted.error_message()));

  Status concat_status = c.as_concatenated_status();
  ASSERT_EQ(concat_status.code(), internal.code());
  ASSERT_TRUE(absl::StrContains(concat_status.error_message(),
                                internal.error_message()));
  ASSERT_TRUE(absl::StrContains(concat_status.error_message(),
                                cancelled.error_message()));
  ASSERT_TRUE(absl::StrContains(concat_status.error_message(),
                                aborted.error_message()));
}

static void BM_TF_CHECK_OK(int iters) {
  tensorflow::Status s =
      (iters < 0) ? errors::InvalidArgument("Invalid") : Status::OK();
  for (int i = 0; i < iters; i++) {
    TF_CHECK_OK(s);
  }
}
BENCHMARK(BM_TF_CHECK_OK);

}  // namespace tensorflow
