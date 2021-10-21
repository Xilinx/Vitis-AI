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
#ifndef _DUP_MATCH_COMPOUND_PREDICATE_HPP_
#define _DUP_MATCH_COMPOUND_PREDICATE_HPP_

#include "dm/common.hpp"

namespace dup_match {
namespace internal {

template <class Predicate1, class Predicate2>
class CompoundPredicate {
   public:
    CompoundPredicate(std::vector<std::string>& column1,
                      std::vector<std::string>& column2,
                      std::vector<std::string>& blockKey) {
        std::vector<std::string> block_key1;
        std::vector<std::string> block_key2;
        Predicate1 pred1(column1, block_key1);
        Predicate2 pred2(column2, block_key2);
        for (uint32_t i = 0; i < column1.size(); i++) {
            if (block_key1[i].size() > 0 && block_key2[i].size() > 0) {
                std::string str = block_key1[i] + block_key2[i];
                blockKey.push_back(str);
            }
        }
    }

    CompoundPredicate(int t,
                      std::vector<std::string>& column1,
                      std::vector<std::string>& column2,
                      std::vector<std::string>& blockKey) {
        std::vector<std::vector<std::string> > block_key1(column1.size());
        std::vector<uint32_t> index_id2(column2.size());
        Predicate1 pred1(column1, block_key1);
        Predicate2 pred2(column2, index_id2);
        for (uint32_t i = 0; i < column1.size(); i++) {
            if (block_key1[i].size() > 0 && index_id2[i] != -1) {
                for (int j = 0; j < block_key1[i].size(); j++) {
                    std::string str = block_key1[i][j] + std::to_string(index_id2[i]);
                    blockKey.push_back(str);
                }
            }
        }
    }

    CompoundPredicate(std::string& xclbinPath, std::vector<std::string>& column1, std::vector<std::string>& column2) {
        index_id1_[0] = aligned_alloc<uint32_t>(TwoGramPredicate::RN);
        index_id1_[1] = aligned_alloc<uint32_t>(TwoGramPredicate::RN);
        pred1 = new Predicate1(xclbinPath, column1, index_id1_);
        index_id2_.resize(column2.size());
        Predicate2 pred2(column2, index_id2_);
    }

    void finish(std::vector<std::string>& blockKey) {
        pred1->finish();
        for (int i = 0; i < index_id2_.size() / CU; i++) {
            if (index_id1_[0][i] != -1 && index_id2_[i] != -1) {
                std::string str = std::to_string(index_id1_[0][i]) + std::to_string(index_id2_[i]);
                // std::cout << "1: str[" << i << "]=" << str << std::endl;
                blockKey.push_back(str);
            }
        }
        for (int i = index_id2_.size() / CU; i < index_id2_.size(); i++) {
            if (index_id1_[1][i - index_id2_.size() / CU] != -1 && index_id2_[i] != -1) {
                std::string str =
                    std::to_string(index_id1_[1][i - index_id2_.size() / CU]) + std::to_string(index_id2_[i]);
                // std::cout << "2: str[" << i << "]=" << str << std::endl;
                blockKey.push_back(str);
            }
        }
    }

   private:
    Predicate1* pred1;
    uint32_t* index_id1_[2];
    std::vector<uint32_t> index_id2_;
};

} // internal
} // dup_match
#endif
