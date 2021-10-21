// Copyright 2019 Google LLC
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#include "pik/cmdline.h"

#include <string>

namespace pik {
namespace tools {

void CommandLineParser::PrintHelp() const {
  fprintf(stderr, "Usage: %s [OPTIONS]\n",
          program_name_ ? program_name_ : "command");
  for (const auto& option : options_) {
    fprintf(stderr, " %s\n", option->help_flags().c_str());
    const char* help_text = option->help_text();
    if (help_text) {
      fprintf(stderr, "    %s\n", help_text);
    }
  }
  fprintf(stderr, " --help\n    Prints this help message.\n");
}

bool CommandLineParser::Parse(int argc, const char* argv[]) {
  if (argc) program_name_ = argv[0];
  int i = 1;  // argv[0] is the program name.
  while (i < argc) {
    if (!strcmp("--help", argv[i])) {
      // Returning false on Parse() forces to print the help message.
      return false;
    }
    bool found = false;
    for (const auto& option : options_) {
      if (option->Match(argv[i])) {
        // Parsing advances the value i on success.
        const char* arg = argv[i];
        if (!option->Parse(argc, argv, &i)) {
          fprintf(stderr, "Error parsing flag %s\n", arg);
          return false;
        }
        found = true;
        break;
      }
    }
    if (!found) {
      // No option matched argv[i].
      fprintf(stderr, "Unknown argument: %s\n", argv[i]);
      return false;
    }
  }
  return true;
}

}  // namespace tools
}  // namespace pik
