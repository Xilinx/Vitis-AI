/*
 * Copyright 2019 Xilinx, Inc.
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
/*
 * Copyright 2019 Xilinx, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef XF_BLAS_HANDLE_HPP
#define XF_BLAS_HANDLE_HPP

#include <vector>
#include <memory>

#include <unordered_map>
#include "host.hpp"

using namespace std;

namespace xf {

namespace blas {

class BLASHostHandle {
   public:
    unordered_map<unsigned int, vector<shared_ptr<BLASHost> > > m_handlePtr;
    static BLASHostHandle& instance() {
        static BLASHostHandle theInstance;
        return theInstance;
    }

   protected:
    BLASHostHandle() {}
};

class ConfigDict {
   public:
    unordered_map<string, string> m_dict = {};
    static ConfigDict& instance() {
        static ConfigDict theInstance;
        return theInstance;
    }

   protected:
    ConfigDict() {}
};

} // namespace blas

} // namespace xf

#endif