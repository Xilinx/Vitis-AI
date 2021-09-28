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

#include "dup_match.hpp"

DupMatch::DupMatch(const std::string& inFile,
                   const std::string& goldenFile,
                   const std::vector<std::string>& fieldName,
                   const std::string& xclbinPath) {
    columns_.resize(fieldName.size());
    // read csv file
    rapidcsv::Document doc(inFile.c_str(), rapidcsv::LabelParams(0, -1),
                           rapidcsv::SeparatorParams(',', true, rapidcsv::sPlatformHasCR, true, true));
    for (int i = 0; i < fieldName.size(); i++) columns_[i] = doc.GetColumn<std::string>(fieldName[i]);
    for (int i = 0; i < fieldName.size(); i++) {
        for (int j = 0; j < columns_[i].size(); j++) {
            columns_[i][j] = dup_match::internal::to_preprocess(columns_[i][j]);
        }
    }
    xclbin_path_ = xclbinPath;
}

void DupMatch::run(std::vector<std::pair<uint32_t, double> >& clusterMembership) {
    std::cout << "DupMatch::run...\n";
    using namespace dup_match::internal;
    // compound predicate, get block
    CompoundPredicate<TwoGramPredicate, WordPredicate> tw_preds(xclbin_path_, columns_[0], columns_[1]);
    CompoundPredicate<AlphaNumPredicate, WordPredicate> aw_preds(0, columns_[0], columns_[0], block_key_[1]);
    CompoundPredicate<StringPredicate, SimplePredicate> ss_preds(columns_[0], columns_[2], block_key_[2]);
    // Kernel finish
    tw_preds.finish(block_key_[0]);

    Cluster cluster = Cluster();
    // dedupe pair
    cluster.pairWrap(block_key_);
    // score by weighted
    cluster.score(columns_);
    // CC to find sub-graph
    cluster.connectedComponents();
    clusterMembership.resize(columns_[0].size(), std::pair<uint32_t, double>(-1, 0.0));
    // hierarchical cluster
    cluster.hierCluster(clusterMembership);
    std::cout << "DupMatch::run End\n";
}
