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

#include "cluster.hpp"

namespace dup_match {
namespace internal {

void Cluster::pair(std::vector<std::string>& blockKey, std::set<std::pair<uint32_t, uint32_t> >& pairId) {
    std::vector<std::pair<std::string, uint32_t> > bk_id(blockKey.size());
    for (uint32_t i = 0; i < blockKey.size(); i++) {
        bk_id[i] = std::pair<std::string, uint32_t>(blockKey[i], i);
    }

    std::sort(bk_id.begin(), bk_id.end());

    std::vector<uint32_t> vec_id;
    std::string str_last = "";
    bk_id.push_back(std::pair<std::string, uint32_t>("a", -1)); // for below loop
    for (uint32_t i = 0; i < bk_id.size(); i++) {
        if (str_last != bk_id[i].first) {
            for (int j = 0; j < vec_id.size(); j++) {
                for (int k = j + 1; k < vec_id.size(); k++) {
                    pairId.insert(std::pair<uint32_t, uint32_t>(vec_id[j], vec_id[k]));
                }
            }
            vec_id.clear();
            str_last = bk_id[i].first;
        }
        vec_id.push_back(bk_id[i].second);
    }
    std::cout << "CompoundPredicate: pair size=" << pairId.size() << std::endl;
}

double Cluster::predictProba(std::array<double, 6>& scores) {
    double r = 0.0;
    for (int i = 0; i < scores.size(); i++) {
        r += weights_[i] * scores[i];
    }
    r = std::exp(r + bias_) / (1 + std::exp(r + bias_));
    return r;
}

void Cluster::score(std::vector<std::vector<std::string> >& vec_fields) {
    std::array<double, 6> distances;
    scores_.resize(pair_id_.size());
    for (int i = 0; i < pair_id_.size(); i++) {
        for (int j = 0; j < vec_fields.size(); j++) {
            std::string str1 = vec_fields[j][pair_id_[i].first];
            std::string str2 = vec_fields[j][pair_id_[i].second];
            if ((j == 3 || j == 2) && (str1.size() == 0 || str2.size() == 0)) {
                distances[j] = 0.0;
                distances[j + 2] = 0.0;
            } else {
                distances[j] = ext::distance(str1, str2, j == 2);
                if (j == 3 || j == 2) distances[j + 2] = 1.0;
            }
        }
        scores_[i] = predictProba(distances);
    }
}

void Cluster::pairWrap(std::array<std::vector<std::string>, 3>& blockKey) {
    std::set<std::pair<uint32_t, uint32_t> > set_pair_id;
    for (int i = 0; i < blockKey.size(); i++) {
        pair(blockKey[i], set_pair_id);
    }
    pair_id_.resize(set_pair_id.size());
    int pi = 0;
    for (std::set<std::pair<uint32_t, uint32_t> >::iterator iter = set_pair_id.begin(); iter != set_pair_id.end();
         iter++) {
        pair_id_[pi++] = *iter;
    }
}

#define size_(r_) (((r_ < n) ? 1 : linkage[r_ - n].label))

void unionFind(std::vector<std::pair<uint32_t, uint32_t> >& pairId, std::vector<uint32_t>& labels) {
    std::map<uint32_t, uint32_t> root;
    // std::map<uint32_t, std::vector<uint32_t> > components;
    std::vector<std::vector<uint32_t> > components(pairId.size());
    for (int i = 0; i < pairId.size(); i++) {
        uint32_t a = pairId[i].first;
        uint32_t b = pairId[i].second;
        std::map<uint32_t, uint32_t>::iterator iter_a = root.find(a);
        std::map<uint32_t, uint32_t>::iterator iter_b = root.find(b);
        if (iter_a == root.end() && iter_b == root.end()) {
            root[a] = i;
            root[b] = i;
            components[i].push_back(i);
        } else if (iter_a == root.end()) {
            uint32_t root_b = iter_b->second;
            root[a] = root_b;
            components[root_b].push_back(i);
        } else if (iter_b == root.end()) {
            uint32_t root_a = iter_a->second;
            root[b] = root_a;
            components[root_a].push_back(i);
        } else if (iter_a->second != iter_b->second) {
            // std::cout << "iter_a.second=" << iter_a->second << ", iter_b.second=" << iter_b->second << std::endl;
            uint32_t root_a = iter_a->second;
            uint32_t root_b = iter_b->second;
            if (components[root_a].size() < components[root_b].size()) {
                root_a = iter_b->second;
                root_b = iter_a->second;
            }
            // for(int j=0;j<components[root_b].size();j++){
            //    components[root_a].push_back(components[root_b][j]);
            //}
            components[root_a].insert(components[root_a].end(), components[root_b].begin(), components[root_b].end());
            components[root_a].push_back(i);
            std::set<uint32_t> component_b;
            for (int j = 0; j < components[root_b].size(); j++) {
                component_b.insert(pairId[components[root_b][j]].first);
                component_b.insert(pairId[components[root_b][j]].second);
            }
            for (std::set<uint32_t>::iterator iter = component_b.begin(); iter != component_b.end(); iter++) {
                root[*iter] = root_a;
            }
            components[root_b].clear();
        } else {
            components[iter_a->second].push_back(i);
            continue;
        }
    }
    // int cnt = 0;
    // for (std::map<uint32_t, uint32_t>::iterator iter = root.begin(); iter != root.end(); iter++) {
    //    std::cout << "root[" << cnt++ << "] = (" << iter->first << " ," << iter->second << ")\n";
    //}
    for (int i = 0; i < components.size(); i++) {
        for (int j = 0; j < components[i].size(); j++) {
            // std::cout << "components[" << i << "][" << j << "]=" << components[i][j] << std::endl;
            labels[components[i][j]] = i;
        }
    }
    // for (int i = 0; i < labels.size(); i++) {
    //    std::cout << "label[" << i << "]=" << labels[i] << std::endl;
    //}
}

bool comparer(GraphNode a, GraphNode b) {
    return a.label < b.label;
}

void Cluster::connectedComponents() {
    std::vector<uint32_t> labels(pair_id_.size());
    graph_.resize(pair_id_.size());
    unionFind(pair_id_, labels);
    for (int i = 0; i < labels.size(); i++) {
        GraphNode gl{pair_id_[i].first, pair_id_[i].second, scores_[i], labels[i]};
        graph_[i] = gl;
    }
    std::sort(graph_.begin(), graph_.end(), comparer);
    // for (int i = 0; i < graph_.size(); i++) {
    //    std::cout << "graph_[" << i << "]={" << graph_[i].label << ", " << graph_[i].node1 << ", " << graph_[i].node2
    //              << ", " << std::setprecision(10) << graph_[i].dist << "}\n";
    //}
}

int condensedDistance(std::vector<GraphNode>& graph_,
                      uint32_t begin,
                      uint32_t end,
                      std::map<uint32_t, uint32_t>& iToId,
                      std::vector<double>& distance) {
    std::set<uint32_t> node_set;
    // std::cout << "begin=" << begin << ", end=" << end << std::endl;
    for (uint32_t i = begin; i < end; i++) {
        GraphNode gl = graph_[i];
        node_set.insert(gl.node1);
        node_set.insert(gl.node2);
    }
    uint32_t cnt = 0;
    std::map<uint32_t, uint32_t> id_to_i;
    for (std::set<uint32_t>::iterator iter = node_set.begin(); iter != node_set.end(); iter++) {
        // std::cout << "node_set=" << *iter << std::endl;
        iToId[cnt] = *iter;
        // std::cout << "iToId[" << cnt << "]=" << iToId[cnt] << std::endl;
        id_to_i[*iter] = cnt++;
    }
    uint32_t mat_len = node_set.size() * (node_set.size() - 1) / 2;
    std::vector<uint32_t> index(end - begin);
    for (uint32_t i = 0; i < end - begin; i++) {
        std::map<uint32_t, uint32_t>::iterator iter1 = id_to_i.find(graph_[i + begin].node1);
        std::map<uint32_t, uint32_t>::iterator iter2 = id_to_i.find(graph_[i + begin].node2);
        uint32_t row_step = (node_set.size() - iter1->second) * (node_set.size() - iter1->second - 1) / 2;
        index[i] = mat_len - row_step + iter2->second - iter1->second - 1;
        // std::cout << "graph_[" << i << "]=(" << graph_[i + begin].node1 << ", " << graph_[i + begin].node2 << "), ("
        //          << iter1->second << ", " << iter2->second << "), row_step=" << row_step << ", index=" << index[i]
        //          << std::endl;
    }
    distance.resize(mat_len);
    for (int i = 0; i < mat_len; i++) distance[i] = 1.0;

    for (int i = 0; i < index.size(); i++) {
        distance[index[i]] = 1 - graph_[begin + i].dist;
    }
    // for (int i = 0; i < mat_len; i++) {
    //    std::cout << "condensedDistance[" << i << "]=" << distance[i] << std::endl;
    //}
    return node_set.size();
}

void fcluster(std::vector<double>& distance, std::vector<GraphNode>& linkage) {
    int n = int(std::ceil(std::sqrt(distance.size() * 2)));
    if (int(n * (n - 1) / 2) != distance.size()) {
        std::cout << "[ERROR] The length of the condensed distance matrix must be (k \choose 2) for k data points!\n";
        exit(1);
    }
    linkage.resize(n - 1);
    std::vector<t_index> members(n, 1);
    cluster_result cr(n - 1);
    for (t_float* DD = distance.data(); DD != distance.data() + n * (n - 1) / 2; ++DD) *DD *= *DD;
    generic_linkage<METHOD_METR_CENTROID, t_index>(n, distance.data(), members.data(), cr);
    for (int i = 0; i < n - 1; i++) {
        node* nv = cr[i]; // node value
        if (nv->node1 < nv->node2) {
            linkage[i].node1 = nv->node1;
            linkage[i].node2 = nv->node2;
        } else {
            linkage[i].node1 = nv->node2;
            linkage[i].node2 = nv->node1;
        }
        linkage[i].dist = std::sqrt(nv->dist);
        linkage[i].label = size_(nv->node1) + size_(nv->node2); // size
        // std::cout << "linkage[" << i << "]=(" << linkage[i].node1 << ", " << linkage[i].node2 << ", " <<
        // linkage[i].dist
        //          << ", " << linkage[i].label << ")\n";
    }
}

// hierarchical Cluster
void Cluster::hierCluster(std::vector<std::pair<uint32_t, double> >& membership) {
    GraphNode gl{0, 0, 0.0, (uint32_t)-1};
    graph_.push_back(gl); // for end
    uint32_t begin = 0;
    GraphNode last_graph = graph_[0];
    uint32_t cluster_id = 0;
    for (int i = 0; i < graph_.size(); i++) {
        GraphNode now_graph = graph_[i];
        if (now_graph.label != last_graph.label) {
            // std::cout << "\n\n---------- cluster_id=" << cluster_id << std::endl;
            if (i - begin > 1) {
                std::vector<double> condensed_distance;
                std::map<uint32_t, uint32_t> i_to_id;
                i_to_id.clear();
                int n = condensedDistance(graph_, begin, i, i_to_id, condensed_distance);
                std::vector<GraphNode> linkage;
                // fastcluster.linkage
                fcluster(condensed_distance, linkage);
                std::vector<int> partition(linkage.size() + 1, 0);
                // hcluster.fcluster
                cluster_dist(linkage.data(), partition.data(), 0.5, partition.size());
                uint32_t max_id = 0;
                for (int j = 0; j < partition.size(); j++) {
                    if (max_id < partition[j]) max_id = partition[j];
                }
                std::vector<std::vector<uint32_t> > clusters(max_id + 1);
                for (int j = 0; j < partition.size(); j++) {
                    clusters[partition[j]].push_back(j);
                }
                for (int j = 0; j < clusters.size(); j++) {
                    if (clusters[j].size() > 1) {
                        std::vector<double> score_d(clusters[j].size(), 0.0);
                        for (int k1 = 0; k1 < clusters[j].size(); k1++) {
                            for (int k2 = k1 + 1; k2 < clusters[j].size(); k2++) {
                                uint32_t x = clusters[j][k1];
                                uint32_t y = clusters[j][k2];
                                uint32_t index = n * (n - 1) / 2 - (n - x) * (n - x - 1) / 2 + y - x - 1;
                                double squared_dist = condensed_distance[index] * condensed_distance[index];
                                score_d[k1] += squared_dist;
                                score_d[k2] += squared_dist;
                            }
                        }
                        for (int k = 0; k < score_d.size(); k++) {
                            score_d[k] /= score_d.size() - 1;
                            score_d[k] = 1 - std::sqrt(score_d[k]);
                            // std::cout << "score_d[" << k << "]=" << score_d[k] << std::endl;
                            membership[i_to_id[clusters[j][k]]] = std::pair<uint32_t, double>(cluster_id, score_d[k]);
                        }
                        cluster_id++;
                    } else if (clusters[j].size() == 1) {
                        membership[i_to_id[clusters[j][0]]] = std::pair<uint32_t, double>(cluster_id++, 1.0);
                    }
                }
            } else {
                if (graph_[begin].dist > 0.5) {
                    membership[graph_[begin].node1] = std::pair<uint32_t, double>(cluster_id, graph_[begin].dist);
                    membership[graph_[begin].node2] = std::pair<uint32_t, double>(cluster_id++, graph_[begin].dist);
                    // std::cout << "alone node-pair=" << graph_[begin].node1 << ", " << graph_[begin].node2 <<
                    // std::endl;
                }
            }

            begin = i;
            last_graph = now_graph;
        }
    }
    for (int i = 0; i < membership.size(); i++) {
        if (membership[i].first == uint32_t(-1)) membership[i].first = cluster_id++;
        // std::cout << "membership[" << i << "]=(" << membership[i].first << ", " << membership[i].second << ")"
        //          << std::endl;
    }
    std::cout << "duplicate sets " << cluster_id << std::endl;
}

} // internal
} // dup_match
