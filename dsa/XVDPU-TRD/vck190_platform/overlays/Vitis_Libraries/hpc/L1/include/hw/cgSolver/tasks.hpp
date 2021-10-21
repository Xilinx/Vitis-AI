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

#ifndef XF_HPC_CG_TASKS_HPP
#define XF_HPC_CG_TASKS_HPP

namespace xf {
namespace hpc {
namespace cg {

template <typename t_DataType>
class Task {
   public:
    Task(uint32_t p_maxIter = 4096, t_DataType p_tol = 1e-8) {
        m_maxIter = p_maxIter;
        m_tols = p_tol;
        m_iter = 0;
        m_res = 0;
    }

    void setRes(t_DataType p_res) { m_res = p_res; }
    void setMaxIter(uint32_t p_maxIter) { m_maxIter = p_maxIter; }
    void setTols(t_DataType p_tol) { m_tols = p_tol; }
    void setIter(uint32_t p_iter) { m_iter = p_iter; }

    t_DataType getTols() const { return m_tols; }
    uint32_t getIter() const { return m_iter; }
    t_DataType getRes() const { return m_res; }

    bool increase() {
        m_iter++;
        return m_iter < m_maxIter;
    }
    bool meetTol() { return m_res < m_tols; }

   private:
    uint32_t m_maxIter, m_iter;
    t_DataType m_tols, m_res;
};
}
}
}
#endif
