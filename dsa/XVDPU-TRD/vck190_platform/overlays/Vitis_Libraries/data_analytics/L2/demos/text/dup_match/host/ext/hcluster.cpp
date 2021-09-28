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

/*
 * Copyright (C) 2007-2014 Damian Eads
 * All rights reserved
 *
 * Copyright (c) 2001, 2002 Enthought, Inc.
 * All rights reserved.
 *
 * Copyright (c) 2003-2012 SciPy Developers.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *   a. Redistributions of source code must retain the above copyright notice,
 *      this list of conditions and the following disclaimer.
 *   b. Redistributions in binary form must reproduce the above copyright
 *      notice, this list of conditions and the following disclaimer in the
 *      documentation and/or other materials provided with the distribution.
 *   c. Neither the name of Enthought nor the names of the SciPy Developers
 *      may be used to endorse or promote products derived from this software
 *      without specific prior written permission.
 *
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS
 * BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY,
 * OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
 * THE POSSIBILITY OF SUCH DAMAGE.
 */

/*
 * This file comes from open source dedupeio/hcluster: https://github.com/dedupeio/hcluster.
 * This file is used by XILINX.
 * In order to be called, convert it into C++ language by XILINX.
 */

#include <vector>
#include <cstring>
#include <iostream>
#include "ext.hpp"

inline int is_visited(uint8_t* bitset, int i) {
    /*
    Check if node i was visited.
    */
    return bitset[i >> 3] & (1 << (i & 7));
}

inline void set_visited(uint8_t* bitset, int i) {
    /*
    Mark node i as visited.
    */
    bitset[i >> 3] |= 1 << (i & 7);
}

void get_max_dist_for_each_cluster(GraphNode* Z, std::vector<double>& MD, int n) {
    /*
    Get the maximum inconsistency coefficient for each non-singleton cluster.

    Parameters
    ----------
    Z : ndarray
        The linkage matrix.
    MD : ndarray
        The array to store the result.
    n : int
        The number of observations.
    */
    int k = 0, i_lc, i_rc, root;
    double max_dist, max_l, max_r;
    std::vector<int> curr_node(n);

    int visited_size = (((n * 2) - 1) >> 3) + 1;
    uint8_t* visited = (uint8_t*)malloc(visited_size);
    // std::cout << "visited_size=" << visited_size << std::endl;
    std::memset(visited, 0, visited_size);

    curr_node[0] = 2 * n - 2;
    while (k >= 0) {
        root = curr_node[k] - n;
        // std::cout << "root=" << root << ", k=" << k << ", visited=" << (int)visited[0] << std::endl;
        i_lc = (int)Z[root].node1;
        i_rc = (int)Z[root].node2;
        // std::cout << "i_lc=" << i_lc << ", i_rc=" << i_rc << std::endl;

        if (i_lc >= n && !is_visited(visited, i_lc)) {
            set_visited(visited, i_lc);
            k += 1;
            curr_node[k] = i_lc;
            continue;
        }

        if (i_rc >= n && !is_visited(visited, i_rc)) {
            set_visited(visited, i_rc);
            k += 1;
            curr_node[k] = i_rc;
            continue;
        }

        max_dist = Z[root].dist;
        if (i_lc >= n) {
            max_l = MD[i_lc - n];
            if (max_l > max_dist) {
                max_dist = max_l;
            }
        }
        if (i_rc >= n) {
            max_r = MD[i_rc - n];
            if (max_r > max_dist) {
                max_dist = max_r;
            }
        }
        MD[root] = max_dist;

        k -= 1;
    }
    free(visited);
}

void cluster_monocrit(GraphNode* Z, std::vector<double>& MC, int* T, double cutoff, int n) {
    /*
    Form flat clusters by monocrit criterion.

    Parameters
    ----------
    Z : ndarray
        The linkage matrix.
    MC : ndarray
        The monotonic criterion array.
    T : ndarray
        The array to store the cluster numbers. The i'th observation belongs to
        cluster `T[i]`.
    cutoff : double
        Clusters are formed when the MC values are less than or equal to
        `cutoff`.
    n : int
        The number of observations.
    */
    int k, i_lc, i_rc, root, n_cluster = 0, cluster_leader = -1;
    std::vector<int> curr_node(n);

    int visited_size = (((n * 2) - 1) >> 3) + 1;
    uint8_t* visited = (uint8_t*)malloc(visited_size);
    std::memset(visited, 0, visited_size);

    k = 0;
    curr_node[0] = 2 * n - 2;
    while (k >= 0) {
        root = curr_node[k] - n;
        i_lc = (int)Z[root].node1;
        i_rc = (int)Z[root].node2;

        if (cluster_leader == -1 && MC[root] <= cutoff) { // found a cluster
            cluster_leader = root;
            n_cluster += 1;
        }

        if (i_lc >= n && !is_visited(visited, i_lc)) {
            set_visited(visited, i_lc);
            k += 1;
            curr_node[k] = i_lc;
            continue;
        }

        if (i_rc >= n && !is_visited(visited, i_rc)) {
            set_visited(visited, i_rc);
            k += 1;
            curr_node[k] = i_rc;
            continue;
        }

        if (i_lc < n) {
            if (cluster_leader == -1) { // singleton cluster
                n_cluster += 1;
            }
            T[i_lc] = n_cluster;
        }

        if (i_rc < n) {
            if (cluster_leader == -1) { // singleton cluster
                n_cluster += 1;
            }
            T[i_rc] = n_cluster;
        }

        if (cluster_leader == root) { // back to the leader
            cluster_leader = -1;
        }
        k -= 1;
    }
    free(visited);
}

void cluster_dist(GraphNode* Z, int* T, double cutoff, int n) {
    // std::cout << "hcluster...\n";
    /*
     Form flat clusters by distance criterion.

     Parameters
     ----------
     Z : ndarray
         The linkage matrix.
     T : ndarray
         The array to store the cluster numbers. The i'th observation belongs to
         cluster `T[i]`.
     cutoff : double
         Clusters are formed when distances are less than or equal to `cutoff`.
     n : int
         The number of observations.
     */

    // cdef double[:] max_dists = np.ndarray(n, dtype=np.double)
    std::vector<double> max_dists(n);
    get_max_dist_for_each_cluster(Z, max_dists, n);
    cluster_monocrit(Z, max_dists, T, cutoff, n);
}
