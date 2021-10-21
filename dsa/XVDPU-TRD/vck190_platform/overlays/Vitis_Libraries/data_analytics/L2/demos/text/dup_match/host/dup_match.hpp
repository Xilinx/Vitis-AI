/*
 * Copyright 2021 Xilinx, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef _DUP_MATCH_HPP_
#define _DUP_MATCH_HPP_

#include <iostream>
#include "dm/predicate.hpp"
#include "dm/compound_predicate.hpp"
#include "dm/cluster.hpp"
#include "ext/ext.hpp"

class DupMatch {
   public:
    DupMatch(const std::string& inFile,
             const std::string& goldenFile,
             const std::vector<std::string>& fieldName,
             const std::string& xclbinPath);
    void run(std::vector<std::pair<uint32_t, double> >& clusterMembership);

   private:
    std::string xclbin_path_;
    std::vector<std::vector<std::string> > columns_;
    std::array<std::vector<std::string>, 3> block_key_;
};

#endif
