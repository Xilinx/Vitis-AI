/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/dump.h"

#include "absl/strings/ascii.h"
#include "tensorflow/compiler/xla/service/hlo_graph_dumper.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_proto_util.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/proto_serialization.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/regexp.h"

namespace xla {

namespace {

using absl::StrCat;
using absl::StrFormat;
using absl::string_view;

struct CanonicalDebugOptions {
  explicit CanonicalDebugOptions(const DebugOptions& opts)
      : dump_to(opts.xla_dump_to()),
        dump_as_text(opts.xla_dump_hlo_as_text()),
        dump_as_proto(opts.xla_dump_hlo_as_proto()),
        dump_as_dot(opts.xla_dump_hlo_as_dot()),
        dump_as_html(opts.xla_dump_hlo_as_html()),
        dump_as_url(opts.xla_dump_hlo_as_url()),
        dump_snapshots(opts.xla_dump_hlo_snapshots()) {
    // This constructor examines the values in `opts` and turns on other flags
    // based on what we think is the user's intent.  To reduce confusion about
    // what was a user-specified value versus an extrapolated value, within this
    // function we treat this struct's members as write-only, and read only from
    // `opts`.

    // Did the user specifiy an explicit format for dumping?
    bool output_format_other_than_url_specified =
        opts.xla_dump_hlo_as_text() || opts.xla_dump_hlo_as_proto() ||
        opts.xla_dump_hlo_as_dot() || opts.xla_dump_hlo_as_html() ||
        opts.xla_dump_hlo_snapshots();
    bool output_format_specified =
        output_format_other_than_url_specified || opts.xla_dump_hlo_as_url();

    // If we haven't specified an output format, default to dumping as text.
    if (!output_format_specified) {
      dump_as_text = true;
    }

    // If dump_to is empty, default to dumping to stdout, so long as some dump
    // format other than dump-as-url was specified.  If the user only specified
    // --xla_dump_hlo_as_url, then don't dump to stdout, that is likely noise
    // they don't want.
    if (opts.xla_dump_to().empty() && output_format_other_than_url_specified) {
      dump_to = "-";
    }

    // If we specified a regular expression restricting which modules to dump,
    // respect that.
    //
    // If we didn't specify which modules to dump but we passed some other flag
    // which implies dumping modules, dump all modules.
    //
    // Otherwise, don't dump any HLO modules.
    if (!opts.xla_dump_hlo_module_re().empty()) {
      // RE2 object is not copyable, and we can't capture "by move", so we
      // resort to this hack.
      string pattern = opts.xla_dump_hlo_module_re();
      should_dump_module = [pattern](string_view module_name) {
        return RE2::PartialMatch(string(module_name), pattern);
      };
    } else if (!opts.xla_dump_hlo_pass_re().empty() ||
               !opts.xla_dump_to().empty() || output_format_specified) {
      should_dump_module = [](string_view) { return true; };
    } else {
      should_dump_module = [](string_view) { return false; };
    }

    // Initialize should_dump_pass.  This one is easy: We only dump per-pass
    // data if the user asked for it explicitly.
    if (!opts.xla_dump_hlo_pass_re().empty()) {
      string pattern = opts.xla_dump_hlo_pass_re();
      should_dump_pass = [pattern](string_view pass_name) {
        return RE2::PartialMatch(string(pass_name), pattern);
      };
    } else {
      should_dump_pass = [](string_view) { return false; };
    }

    // Output dirs "sponge" and "test_undeclared_outputs_dir" (case-insensitive)
    // have a special meaning: Dump into the directory specified by the
    // environment variable TEST_UNDECLARED_OUTPUTS_DIR.
    string dump_to_lower = absl::AsciiStrToLower(opts.xla_dump_to());
    if (dump_to_lower == "sponge" ||
        dump_to_lower == "test_undeclared_outputs_dir") {
      const char* dir = getenv("TEST_UNDECLARED_OUTPUTS_DIR");
      if (dir != nullptr) {
        dump_to = dir;
      } else {
        LOG(ERROR) << "--xla_dump_to=" << opts.xla_dump_to()
                   << ", but environment variable TEST_UNDECLARED_OUTPUTS_DIR "
                      "is not set, so cannot dump anywhere.";
        should_dump_module = [](string_view) { return false; };
        should_dump_pass = [](string_view) { return false; };
      }
    }
  }

  bool dumping_to_stdout() const { return dump_to == "-"; }

  string dump_to;
  std::function<bool(string_view module_name)> should_dump_module;
  std::function<bool(string_view pass_name)> should_dump_pass;

  // dump_ir isn't present here because this file is mostly concerned with
  // dumping HLO.
  bool dump_as_text;
  bool dump_as_proto;
  bool dump_as_dot;
  bool dump_as_html;
  bool dump_as_url;
  bool dump_snapshots;
};

void DumpToFileInDirImpl(string_view filename, string_view contents,
                         const CanonicalDebugOptions& opts) {
  if (opts.dumping_to_stdout()) {
    LOG(ERROR) << "Refusing to write " << filename
               << " to stdout.  Pass --xla_dump_to=<path> to write to a file.";
    return;
  }

  if (opts.dump_to.empty()) {
    return;
  }

  const string& dir = opts.dump_to;
  VLOG(1) << "Dumping " << filename << " to " << dir;

  tensorflow::Env* env = tensorflow::Env::Default();
  // Two threads can race to observe the absence of the dump directory and
  // simultaneously try to create it, causing the "losing" thread to get a
  // "directory already exists" error.  We can work around this by checking
  // again whether the dir exists.
  if (!env->IsDirectory(dir).ok()) {
    auto status = env->RecursivelyCreateDir(dir);
    if (!status.ok() && !env->IsDirectory(dir).ok()) {
      LOG(ERROR) << "Could not create directory " << dir
                 << " for dumping XLA debug data: " << status;
      return;
    }
  }

  string file_path =
      tensorflow::io::JoinPath(dir, SanitizeFileName(string(filename)));
  auto status = tensorflow::WriteStringToFile(env, file_path, contents);
  if (!status.ok()) {
    LOG(ERROR) << "Could not write XLA debug data to " << file_path << ": "
               << status;
  }
}

void DumpToFileInDirOrStdoutImpl(string_view filename, string_view contents,
                                 const CanonicalDebugOptions& opts) {
  // Dump to stdout if that's called for.
  if (opts.dumping_to_stdout()) {
    std::cout << "*** Begin " << filename << " ***\n"
              << contents << "\n*** End " << filename << " ***" << std::endl;
    return;
  }

  // Otherwise, dump to a file.
  DumpToFileInDirImpl(filename, contents, opts);
}

void DumpHloModuleImpl(const HloModule& module,
                       const BufferAssignment* buffer_assn,
                       const HloExecutionProfile* profile, string_view suffix,
                       const CanonicalDebugOptions& opts) {
  string filename = FilenameFor(module, suffix);

  if (opts.dump_as_text) {
    DumpToFileInDirOrStdoutImpl(StrCat(filename, ".txt"), module.ToString(),
                                opts);
    if (buffer_assn) {
      DumpToFileInDirOrStdoutImpl(StrCat(filename, "-buffer-assignment.txt"),
                                  buffer_assn->ToString(), opts);
    }
  }

  if (opts.dump_as_proto) {
    HloProto module_proto =
        buffer_assn ? MakeHloProto(module, *buffer_assn) : MakeHloProto(module);
    string pb;
    if (!tensorflow::SerializeToStringDeterministic(module_proto, &pb)) {
      pb = "Failed to serialize HLO module proto.";
    }
    DumpToFileInDirImpl(StrCat(filename, ".hlo.pb"), pb, opts);
  }

  auto render_graph = [&](RenderedGraphFormat format) {
    StatusOr<string> rendered_graph = RenderGraph(
        *module.entry_computation(),
        /*label=*/filename, module.config().debug_options(), format, profile);
    if (rendered_graph.ok()) {
      return std::move(rendered_graph).ValueOrDie();
    }
    return StrFormat("Error rendering graph: %s",
                     rendered_graph.status().ToString());
  };

  if (opts.dump_as_dot) {
    DumpToFileInDirImpl(StrFormat("%s.dot", filename),
                        render_graph(RenderedGraphFormat::kDot), opts);
  }

  if (opts.dump_as_html) {
    DumpToFileInDirImpl(StrFormat("%s.html", filename),
                        render_graph(RenderedGraphFormat::kHtml), opts);
  }

  // Special case for rendering graphs as URLs.  We'll dump them to a file
  // because why not, but we always log them to stdout as well.
  if (opts.dump_as_url) {
    string url = render_graph(RenderedGraphFormat::kUrl);
    std::cout << filename << " --> " << url << std::endl;
    if (!opts.dumping_to_stdout()) {
      DumpToFileInDirImpl(StrFormat("%s.url", filename), url, opts);
    }
  }
}

static tensorflow::mutex mu(tensorflow::LINKER_INITIALIZED);

// Maps a module's unique ID to a counter indicating how many times we've dumped
// this module during the compilation pipeline.  This lets us keep the filenames
// ordered nicely.
//
// Entries added here leak forever; we have no way to GC them when a module
// dies.  But we only add an entry if dumping is enabled for this module, and
// dumping a module leaks buffer space in stdout or bytes on disk *way* faster
// than this hashtable leaks memory.
static auto& module_id_to_step_number GUARDED_BY(mu) =
    *new absl::flat_hash_map<int64, int64>();

}  // namespace

string FilenameFor(const HloModule& module, string_view suffix) {
  return StrFormat("module_%04d.%s", module.unique_id(), suffix);
}

void DumpToFileInDir(const HloModule& module, string_view suffix,
                     string_view contents) {
  DumpToFileInDirImpl(FilenameFor(module, suffix), contents,
                      CanonicalDebugOptions(module.config().debug_options()));
}

void DumpToFileInDirOrStdout(const HloModule& module, string_view suffix,
                             string_view contents) {
  DumpToFileInDirOrStdoutImpl(
      FilenameFor(module, suffix), contents,
      CanonicalDebugOptions(module.config().debug_options()));
}

void DumpHloModuleIfEnabled(const HloModule& module, string_view name) {
  CanonicalDebugOptions opts(module.config().debug_options());
  if (opts.should_dump_module(module.name())) {
    DumpHloModuleImpl(module, /*buffer_assn=*/nullptr, /*profile=*/nullptr,
                      name, opts);
  }
}
void DumpHloModuleIfEnabled(const HloModule& module,
                            const BufferAssignment& buffer_assn,
                            string_view name) {
  CanonicalDebugOptions opts(module.config().debug_options());
  if (opts.should_dump_module(module.name())) {
    DumpHloModuleImpl(module, &buffer_assn, /*profile=*/nullptr, name, opts);
  }
}

void DumpHloModuleIfEnabled(const HloModule& module,
                            const HloExecutionProfile& profile,
                            string_view name) {
  CanonicalDebugOptions opts(module.config().debug_options());
  if (opts.should_dump_module(module.name())) {
    DumpHloModuleImpl(module, /*buffer_assn=*/nullptr, &profile, name, opts);
  }
}

bool DumpingEnabledForHloModule(string_view hlo_module_name,
                                const DebugOptions& opts) {
  return CanonicalDebugOptions(opts).should_dump_module(hlo_module_name);
}

bool DumpingToStdout(const DebugOptions& opts) {
  return CanonicalDebugOptions(opts).dumping_to_stdout();
}

void DumpHloModuleBetweenPassesIfEnabled(string_view pipeline_name,
                                         string_view before_pass_name,
                                         string_view after_pass_name,
                                         const HloModule& module) {
  CanonicalDebugOptions opts(module.config().debug_options());
  if (!opts.should_dump_module(module.name())) {
    return;
  }

  if (!opts.should_dump_pass(before_pass_name) &&
      !opts.should_dump_pass(after_pass_name)) {
    return;
  }

  int64 step_number;
  {
    tensorflow::mutex_lock lock(mu);
    step_number = module_id_to_step_number[module.unique_id()]++;
  }

  string filename_suffix =
      StrFormat("%04d.%s.after_%s.before_%s", step_number, pipeline_name,
                after_pass_name, before_pass_name);
  DumpHloModuleImpl(module, /*buffer_assn=*/nullptr, /*profile=*/nullptr,
                    filename_suffix, opts);
}

void DumpHloModuleDuringPassIfEnabled(string_view pass_name,
                                      string_view step_name,
                                      const HloModule& module) {
  CanonicalDebugOptions opts(module.config().debug_options());
  if (!opts.should_dump_module(module.name()) ||
      !opts.should_dump_pass(pass_name)) {
    return;
  }

  int64 step_number;
  {
    tensorflow::mutex_lock lock(mu);
    step_number = module_id_to_step_number[module.unique_id()]++;
  }

  string filename_suffix =
      StrFormat("%04d.%s.%s", step_number, pass_name, step_name);
  DumpHloModuleImpl(module, /*buffer_assn=*/nullptr, /*profile=*/nullptr,
                    filename_suffix, opts);
}

void DumpHloSnapshotIfEnabled(const HloModule& module,
                              const HloSnapshot& snapshot) {
  CanonicalDebugOptions opts(module.config().debug_options());
  if (!opts.should_dump_module(module.name()) || !opts.dump_snapshots) {
    return;
  }
  int64 execution_count;
  {
    static auto& module_id_to_execution_count GUARDED_BY(mu) =
        *new absl::flat_hash_map<int64, int64>();
    tensorflow::mutex_lock lock(mu);
    execution_count = module_id_to_execution_count[module.unique_id()]++;
  }
  string filename =
      StrCat(FilenameFor(module, StrFormat("execution_%04d", execution_count)),
             ".hlo_snapshot.pb");
  if (opts.dumping_to_stdout()) {
    LOG(ERROR) << "Refusing to write HLO snapshot proto for " << filename
               << " to stdout.  Pass --xla_dump_to=<path> to write to a file.";
    return;
  }
  string pb;
  if (!tensorflow::SerializeToStringDeterministic(snapshot, &pb)) {
    LOG(ERROR) << "Failed to serialize HLO snapshot proto " << filename;
  }
  DumpToFileInDirImpl(filename, pb, opts);
}

void DumpHloSnapshotIfEnabled(const HloSnapshot& snapshot,
                              const DebugOptions& opts) {
  CanonicalDebugOptions canonical_opts(opts);
  string name = snapshot.hlo().hlo_module().name();
  if (!canonical_opts.should_dump_module(name) ||
      !canonical_opts.dump_snapshots) {
    return;
  }

  // We don't have a unique id for an HloSnapshot, so in this overload we just
  // have to use its name.
  int64 execution_count;
  {
    static auto& module_name_to_execution_count GUARDED_BY(mu) =
        *new absl::flat_hash_map<string, int64>();
    tensorflow::mutex_lock lock(mu);
    execution_count = module_name_to_execution_count[name]++;
  }
  string filename = StrFormat("module_%s.execution_%04d.hlo_snapshot.pb", name,
                              execution_count);
  if (canonical_opts.dumping_to_stdout()) {
    LOG(ERROR) << "Refusing to write HLO snapshot proto for " << filename
               << " to stdout.  Pass --xla_dump_to=<path> to write to a file.";
    return;
  }
  string pb;
  if (!tensorflow::SerializeToStringDeterministic(snapshot, &pb)) {
    LOG(ERROR) << "Failed to serialize HLO snapshot proto " << filename;
  }
  DumpToFileInDirImpl(filename, pb, canonical_opts);
}

}  // namespace xla
