/**
 * Copyright (C) 2020 Xilinx, Inc
 *
 * Licensed under the Apache License, Version 2.0 (the "License"). You may
 * not use this file except in compliance with the License. A copy of the
 * License is located at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 */

#include "defs.h"
#include "partitionLouvain.hpp"
#include "ctrlLV.h"

void FreeG(graphNew*& G) {
    free(G->edgeListPtrs);
    free(G->edgeList);
    free(G);
}

void printG_org(graphNew* G) {
    long vertexNum = G->numVertices;
    long NE = G->numEdges;
    long* vtxPtr = G->edgeListPtrs;
    edge* vtxInd = G->edgeList;
    printf("v=%ld\t; e=%ld \n", vertexNum, NE);
    for (int v = 0; v < vertexNum; v++) {
        long adj1 = vtxPtr[v];
        long adj2 = vtxPtr[v + 1];
        int degree = adj2 - adj1;
        printf("v=%4ld\t adj1=%4ld, adj2=%4ld, degree=%4ld\t", v, adj1, adj2, degree);
        for (int d = 0; d < degree; d++) {
            printf(" %4ld\t", vtxInd[adj1 + d].tail);
        }
        printf("\n");
    }
}

void printG(graphNew* G, long* C, long* M, long star, long end) {
    long NV = G->numVertices;
    long NE = G->numEdges;
    long* vtxPtr = G->edgeListPtrs;
    edge* vtxInd = G->edgeList;
    printf("v=%ld\t; e=%ld \n", NV, NE);
    if (star < 0) star = 0;
    if (end > NV) end = NV;
    // printf("|==C==|==V==|==M==|=OFF=|=Dgr=|\n");
    for (int v = star; v < end; v++) {
        long adj1 = vtxPtr[v];
        long adj2 = vtxPtr[v + 1];
        int degree = adj2 - adj1;
        // printf("c=%4ld : v=%4ld\t adj1=%4ld, adj2=%4ld, degree=%4ld\t",C[v], v, adj1, adj2, degree);
        // printf("c=%4ld : v=%4ld\t adj1=%4ld, adj2=%4ld, degree=%4ld\t",C[v], v, adj1, adj2, degree);
        // printf("c=%-4d : v=%-4d m=%-4d off=%4d, dgr=%-3d\t",C[v], v, M[v], adj1, degree);
        long m = M == NULL ? v : M[v];
        long c = C == NULL ? v : C[v];
        // printf(" c=%-4d, v=%-4d,", c, v, m, adj1, degree);
        if (m < 0)
            printf(" \033[1;31;40mc=%-5d v=%-5d m=%-5d\033[0m", c, v, m);
        else
            printf(" c=%-5d v=%-5d m=%-5d", c, v, m);
        printf(" o=%-5d d=%-4d |", adj1, degree);
        for (int d = 0; d < degree; d++) {
            //\033[1;31;40mERROR\033[0m
            long t = vtxInd[adj1 + d].tail;
            double w = vtxInd[adj1 + d].weight;
            if (M != NULL && M[t] < 0)
                printf("\033[1;31;40m%5d\033[0m\/%1.0f ", t, w);
            else
                printf("%5d\/%1.0f ", t, w);
        }
        printf("\n");
    }
}
void printG(graphNew* G, long* C, long* M, long star, long end, bool isCid, bool isDir, ParLV* p_par, int idx) {
    long NV = G->numVertices;
    long NE = G->numEdges;
    long* vtxPtr = G->edgeListPtrs;
    edge* vtxInd = G->edgeList;
    printf("v=%ld\t; e=%ld \n", NV, NE);
    if (star < 0) star = 0;
    if (end > NV) end = NV;
    // printf("|==C==|==V==|==M==|=OFF=|=Dgr=|\n");
    for (int v = star; v < end; v++) {
        long adj1 = vtxPtr[v];
        long adj2 = vtxPtr[v + 1];
        int degree = adj2 - adj1;
        long m = M == NULL ? v : M[v];
        long c = C == NULL ? v : C[v];
        long c_final = c + p_par->off_lved[idx];
        // printf(" c=%-4d, v=%-4d,", c, v, m, adj1, degree);
        if (m < 0) {
            if (p_par->st_PreMerged == true) c_final = p_par->p_v_new[idx][v];
            printf(" \033[1;31;40mc=%-5d (%-5d)   v=%-5d m=%-5d c(m)=%-5d\033[0m", c, c_final, v, m,
                   p_par->FindC_nhop(m));
        } else
            printf(" c=%-5d (%-5d)   v=%-5d m=%-5d c(m)=%-5d", c, c_final, v, m, c);
        printf(" o=%-5d d=%-4d |", adj1, degree);
        for (int d = 0; d < degree; d++) {
            long t = vtxInd[adj1 + d].tail;
            if (isDir) {
                if (isCid) {
                    if (C[v] < C[t]) continue;
                } else {
                    if (v < t) continue;
                }
            }
            double w = vtxInd[adj1 + d].weight;
            if (M != NULL && M[t] < 0)
                printf("\033[1;31;40m%5d\033[0m\/%1.0f ", isCid ? C[t] : t, w);
            else
                printf("%5d\/%1.0f ", isCid ? C[t] : t, w);
        }
        printf("\n");
    }
}
void printG(graphNew* G, long* C, long* M, long star, long end, bool isCid, bool isDir) {
    long NV = G->numVertices;
    long NE = G->numEdges;
    long* vtxPtr = G->edgeListPtrs;
    edge* vtxInd = G->edgeList;
    printf("v=%ld\t; e=%ld \n", NV, NE);
    if (star < 0) star = 0;
    if (end > NV) end = NV;
    // printf("|==C==|==V==|==M==|=OFF=|=Dgr=|\n");
    for (int v = star; v < end; v++) {
        long adj1 = vtxPtr[v];
        long adj2 = vtxPtr[v + 1];
        int degree = adj2 - adj1;
        // printf("c=%4ld : v=%4ld\t adj1=%4ld, adj2=%4ld, degree=%4ld\t",C[v], v, adj1, adj2, degree);
        // printf("c=%4ld : v=%4ld\t adj1=%4ld, adj2=%4ld, degree=%4ld\t",C[v], v, adj1, adj2, degree);
        // printf("c=%-4d : v=%-4d m=%-4d off=%4d, dgr=%-3d\t",C[v], v, M[v], adj1, degree);
        long m = M == NULL ? v : M[v];
        long c = C == NULL ? v : C[v];

        // printf(" c=%-4d, v=%-4d,", c, v, m, adj1, degree);
        if (m < 0)
            printf(" \033[1;31;40mc=%-5d v=%-5d m=%-5d\033[0m", c, v, m);
        else
            printf(" c=%-5d v=%-5d m=%-5d", c, v, m);
        printf(" o=%-5d d=%-4d |", adj1, degree);
        for (int d = 0; d < degree; d++) {
            long t = vtxInd[adj1 + d].tail;
            if (isDir) {
                if (isCid) {
                    if (C[v] < C[t]) continue;
                } else {
                    if (v < t) continue;
                }
            }
            double w = vtxInd[adj1 + d].weight;
            if (M != NULL && M[t] < 0)
                printf("\033[1;31;40m%5d\033[0m\/%1.0f ", isCid ? C[t] : t, w);
            else
                printf("%5d\/%1.0f ", isCid ? C[t] : t, w);
        }
        printf("\n");
    }
}
void printG(char* name, graphNew* G, long* C, long* M, long star, long end) {
    FILE* fp = fopen(name, "w");
    if (fp == NULL) return;
    long NV = G->numVertices;
    long NE = G->numEdges;
    long* vtxPtr = G->edgeListPtrs;
    edge* vtxInd = G->edgeList;
    fprintf(fp, "v=%ld\t; e=%ld \n", NV, NE);
    if (star < 0) star = 0;
    if (end > NV) end = NV;
    // printf("|==C==|==V==|==M==|=OFF=|=Dgr=|\n");
    for (int v = star; v < end; v++) {
        long adj1 = vtxPtr[v];
        long adj2 = vtxPtr[v + 1];
        int degree = adj2 - adj1;
        // printf("c=%4ld : v=%4ld\t adj1=%4ld, adj2=%4ld, degree=%4ld\t",C[v], v, adj1, adj2, degree);
        // printf("c=%4ld : v=%4ld\t adj1=%4ld, adj2=%4ld, degree=%4ld\t",C[v], v, adj1, adj2, degree);
        // printf("c=%-4d : v=%-4d m=%-4d off=%4d, dgr=%-3d\t",C[v], v, M[v], adj1, degree);
        long m = M == NULL ? v : M[v];
        long c = C == NULL ? v : C[v];
        // printf(" c=%-4d, v=%-4d,", c, v, m, adj1, degree);
        if (m < 0)
            printf(" \033[1;31;40mc=%-5d v=%-5d m=%-5d\033[0m", c, v, m);
        else
            printf(" c=%-5d v=%-5d m=%-5d", c, v, m);
        if (m < 0)
            fprintf(fp, "[c=%-5d v=%-5d m=%-5d]", c, v, m);
        else
            fprintf(fp, " c=%-5d v=%-5d m=%-5d", c, v, m);
        fprintf(fp, " o=%-5d d=%-4d |", adj1, degree);
        for (int d = 0; d < degree; d++) {
            //\033[1;31;40mERROR\033[0m
            long t = vtxInd[adj1 + d].tail;
            double w = vtxInd[adj1 + d].weight;
            if (M != NULL && M[t] < 0)
                fprintf(fp, "[%5d]%1.0f ", t, w);
            else
                fprintf(fp, "%5d\/%1.0f ", t, w);
            if (M != NULL && M[t] < 0)
                printf("\033[1;31;40m%5d\033[0m\/%1.0f ", t, w);
            else
                printf("%5d\/%1.0f ", t, w);
        }
        fprintf(fp, "\n");
    }
    fclose(fp);
}
void printG_NOWeight(graphNew* G, long* C, long* M, long star, long end) {
    long NV = G->numVertices;
    long NE = G->numEdges;
    long* vtxPtr = G->edgeListPtrs;
    edge* vtxInd = G->edgeList;
    printf("v=%ld\t; e=%ld \n", NV, NE);
    if (star < 0) star = 0;
    if (end > NV) end = NV;
    // printf("|==C==|==V==|==M==|=OFF=|=Dgr=|\n");
    for (int v = star; v < end; v++) {
        long adj1 = vtxPtr[v];
        long adj2 = vtxPtr[v + 1];
        int degree = adj2 - adj1;
        // printf("c=%4ld : v=%4ld\t adj1=%4ld, adj2=%4ld, degree=%4ld\t",C[v], v, adj1, adj2, degree);
        // printf("c=%4ld : v=%4ld\t adj1=%4ld, adj2=%4ld, degree=%4ld\t",C[v], v, adj1, adj2, degree);
        // printf("c=%-4d : v=%-4d m=%-4d off=%4d, dgr=%-3d\t",C[v], v, M[v], adj1, degree);
        long m = M == NULL ? v : M[v];
        long c = C == NULL ? v : C[v];
        // printf(" c=%-4d, v=%-4d,", c, v, m, adj1, degree);
        if (m < 0)
            printf(" \033[1;31;40mc=%-5d v=%-5d m=%-5d\033[0m", c, v, m);
        else
            printf(" c=%-5d v=%-5d m=%-5d", c, v, m);
        printf(" o=%-5d d=%-4d |", adj1, degree);
        for (int d = 0; d < degree; d++) {
            //\033[1;31;40mERROR\033[0m
            long t = vtxInd[adj1 + d].tail;
            if (M != NULL && M[t] < 0)
                printf("\033[1;31;40m%5d\033[0m ", t);
            else
                printf("%5d ", t);
        }
        printf("\n");
    }
}
void printG_old2(graphNew* G, long* C, long* M, long star, long end) {
    long NV = G->numVertices;
    long NE = G->numEdges;
    long* vtxPtr = G->edgeListPtrs;
    edge* vtxInd = G->edgeList;
    printf("v=%ld\t; e=%ld \n", NV, NE);
    if (star < 0) star = 0;
    if (end > NV) end = NV;
    for (int v = star; v < end; v++) {
        long adj1 = vtxPtr[v];
        long adj2 = vtxPtr[v + 1];
        int degree = adj2 - adj1;
        printf("c=%-4d : v=%-4d m=%-4d off=%4d, dgr=%-3d\t", C[v], v, M[v], adj1, degree);
        for (int d = 0; d < degree; d++) {
            printf(" %4d  ", vtxInd[adj1 + d].tail);
        }
        printf("\n");
    }
}

void printG(graphNew* G, long* C, long* M) {
    long NV = G->numVertices;
    printG(G, C, M, 0, NV);
}

void printG(graphNew* G, long* C) {
    long NV = G->numVertices;
    printG(G, C, NULL, 0, NV);
}
long CountV(graphNew* G, long star, long end) {
    long NV = G->numVertices;
    long* vtxPtr = G->edgeListPtrs;
    edge* vtxInd = G->edgeList;
    if (star < 0) star = 0;
    if (end > NV) end = NV;
    map<long, long> map_all;
    map<long, long>::iterator iter_all;
    long cnt = 0;
    for (long v = star; v < end; v++) {
        long adj1 = vtxPtr[v];
        long adj2 = vtxPtr[v + 1];
        int degree = adj2 - adj1;
        if (map_all.find(v) == map_all.end()) map_all[cnt] = cnt++;

        for (int d = 0; d < degree; d++) {
            long e = vtxInd[adj1 + d].tail;
            if (map_all.find(e) == map_all.end()) map_all[cnt] = cnt++;
        } // for
    }
    return cnt;
}

long CountV(graphNew* G) {
    long NV = G->numVertices;
    return CountV(G, 0, NV);
}

long CountVGh(graphNew* G, long star, long end) {
    long NV = G->numVertices;
    long* vtxPtr = G->edgeListPtrs;
    edge* vtxInd = G->edgeList;
    if (star < 0) star = 0;
    if (end > NV) end = NV;
    map<long, long> map_all;
    map<long, long>::iterator iter_all;
    long cnt = 0;
    for (long v = star; v < end; v++) {
        long adj1 = vtxPtr[v];
        long adj2 = vtxPtr[v + 1];
        int degree = adj2 - adj1;

        // if(map_all.find(v) == map_all.end())
        //	map_all[cnt] = cnt++;

        for (int d = 0; d < degree; d++) {
            long e = vtxInd[adj1 + d].tail;
            if (e < star || e >= end) {
                if (map_all.find(e) == map_all.end()) map_all[e] = cnt++;
            }
        } // for
    }
    return cnt;
}

long CountVGh(graphNew* G) {
    long NV = G->numVertices;
    return CountVGh(G, 0, NV);
}
void CreateSubG(graphNew* G_src, long start, long size, graphNew* G_sub, long* M_sub2src, long* M_sub2ghost) {
    long end = start + size;
    long NV_s = G_src->numVertices;
    long NE_s = G_src->numEdges;
    long* off_s = G_src->edgeListPtrs;
    edge* edge_s = G_src->edgeList;

    map<long, long> clusterLocalMap; // Map each neighbor's cluster to a local number
    map<long, long>::iterator storedAlready;
    long numUniqueClusters = 0;

    long num_v_par = 0;
    long num_e_par = 0;
    long v = start;
    map<long, long> map_ghost;
    map<long, long> map_all;
    map<long, long>::iterator iter_ghost;
    map<long, long>::iterator iter_all;

    while (1) {
        long adj1 = off_s[v];
        long adj2 = off_s[v + 1];
        int dgr = adj2 - adj1;
        iter_all = map_all.find(v);
    }

    for (long v = start; v < end; v++) {
        long adj1 = off_s[v];
        long adj2 = off_s[v + 1];
        int degree = adj2 - adj1;
        for (long e = 0; e < degree; e++) {
            long head = edge_s[v + e].head;
            long tail = edge_s[v + e].tail;
            bool isHeadin = (head >= start) && (head < end);
            bool isTailin = (tail >= start) && (tail < end);
            bool isEgAllIn = isHeadin & isTailin;
            bool isEgAllOut = (~isHeadin) & (~isTailin);
            bool isOnlyHeadIn = isHeadin & (~isTailin);
            bool isOnlyTailIn = isTailin & (~isHeadin);
            long head_m;
            long tail_m;

            // Remap the head and tail
            if (isEgAllOut) {
                continue;
            } else {
                storedAlready = clusterLocalMap.find(head);
                if (storedAlready != clusterLocalMap.end()) {
                    head_m = storedAlready->second; // Renumber the cluster id
                } else {
                    clusterLocalMap[head] = numUniqueClusters; ////Does not exist, add to the map
                    head_m = numUniqueClusters;                // Renumber the cluster id
                    numUniqueClusters++;                       // Increment the number
                }
                storedAlready = clusterLocalMap.find(tail);
                if (storedAlready != clusterLocalMap.end()) {
                    tail_m = storedAlready->second; // Renumber the cluster id
                } else {
                    clusterLocalMap[tail] = numUniqueClusters; ////Does not exist, add to the map
                    tail_m = numUniqueClusters;                // Renumber the cluster id
                    numUniqueClusters++;                       // Increment the number
                }
            }
            edge_s[v + e].head = head_m;
            edge_s[v + e].tail = tail_m;
        }
    }
}

void CopyG(graphNew* G_scr, graphNew* G_des) {
    long NV = G_scr->numVertices;
    long NE = G_scr->numEdges;
    G_des->numEdges = NE;
    G_des->numVertices = NV;

    long* vtxPtr = G_scr->edgeListPtrs;
    edge* vtxInd = G_scr->edgeList;
    long* vtxPtr2 = G_des->edgeListPtrs;
    edge* vtxInd2 = G_des->edgeList;

    for (int v = 0; v < NV; v++) {
        long adj1 = vtxPtr[0 + v];
        long adj2 = vtxPtr[0 + v + 1];
        vtxPtr2[v] = adj1 - 0;
        vtxPtr2[v + 1] = adj2 - 0;
        int degree = adj2 - adj1;
        long adj1_des = vtxPtr2[v];
        for (int d = 0; d < degree; d++) {
            vtxInd2[adj1_des + d].head = vtxInd[adj1 + d].head;
            vtxInd2[adj1_des + d].tail = vtxInd[adj1 + d].tail;
            vtxInd2[adj1_des + d].weight = vtxInd[adj1 + d].weight;
        }
    }
}

graphNew* CloneGbad(graphNew* G_scr) {
    long NV = G_scr->numVertices;
    long NE = G_scr->numEdges;
    graphNew* G_des = (graphNew*)malloc(sizeof(graphNew));
    G_des->edgeListPtrs = (long*)malloc(sizeof(long) * (NV + 1));
    G_des->edgeList = (edge*)malloc(sizeof(edge) * NE);
    CopyG(G_scr, G_des);
    return G_des;
}

void CreatSubG(long head, long end_line, graphNew* G_scr, graphNew* G_des) {
    long NV = G_scr->numVertices;
    long* vtxPtr = G_scr->edgeListPtrs;
    edge* vtxInd = G_scr->edgeList;
    long NE = G_scr->numEdges;
    G_des->numVertices = G_scr->numVertices;
    G_des->numEdges = G_scr->numEdges;
    G_des->edgeListPtrs = (long*)malloc(sizeof(long) * (NV + 1));
    long base_edge = vtxPtr[head];
    long size_edge = vtxPtr[head + end_line] - base_edge;
    G_des->edgeList = (edge*)malloc(sizeof(edge) * size_edge);
    long* vtxPtr2 = G_des->edgeListPtrs;
    edge* vtxInd2 = G_des->edgeList;

    for (int v = 0; v < NV; v++) {
        long adj1 = vtxPtr[v];
        long adj2 = vtxPtr[v + 1];
        int degree = adj2 - adj1;
        if (v >= head && v < (head + end_line)) {
            vtxPtr2[v] = adj1 - base_edge;     // txPtr[head]
            vtxPtr2[v + 1] = adj2 - base_edge; // txPtr[head]
            long adj1_des = vtxPtr2[v];
            for (int d = 0; d < degree; d++) {
                vtxInd2[adj1_des + d].head = vtxInd[adj1 + d].head;
                vtxInd2[adj1_des + d].tail = vtxInd[adj1 + d].tail;
                vtxInd2[adj1_des + d].weight = vtxInd[adj1 + d].weight;
            }
        } else if (v < head) {
            vtxPtr2[v] = 0;
            vtxPtr2[v + 1] = 0;
        } else {
            vtxPtr2[v] = size_edge;
            vtxPtr2[v + 1] = size_edge;
        }
    }
}

graphNew* CloneG(graphNew* G_scr) {
    long NV = G_scr->numVertices;
    graphNew* G_des = (graphNew*)malloc(sizeof(graphNew));
    CreatSubG(0, NV, G_scr, G_des);
    return G_des;
}

void InitC(long* C, long NV) {
    assert(C);
    for (int i = 0; i < NV; i++) C[i] = i;
}

/////////////////////////////////////////////////////////////////////////////////////
/// GLV//////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////
GLV::GLV(int& id) {
    InitVar();
    ID = id;
    sprintf(name, "%d\0", id);
    id++;
}

GLV::~GLV() {
    FreeMem();
}

void GLV::InitVar() {
    G = 0;
    C = 0;
    M = 0;
    colors = 0;
    NC = NElg = NVl = 0;
    NV = -1;
    NE = -1;
    numColors = -1;
    numThreads = 1;
    times.parNo = -1; // No partition
};
void GLV::FreeMem() {
    if (G) FreeG(G);
    if (C) free(C);
    if (M) free(M);
    if (colors) free(colors);
}
void GLV::CleanCurrentG() {
    if (G != 0) {
        printf("GLV: Current G is not empty!!!: \n");
        printf("GLV: To be clean G in GLV: ");
        printSimple();
        // displayGraphCharacteristics(G);
        printf("GLV: FreeMem\n");
        FreeMem();
        printf("GLV: InitVar\n");
        InitVar();
    }
}

void GLV::InitByFile(char* name_file) {
    double totTimeColoring;
    CleanCurrentG();
    printf("GLV: host_PrepareGraph(3, %s, 0)\n", name_file);
    G = host_PrepareGraph(3, name_file, 0);
    SyncWithG();
    InitM();
    printf("GLV: displayGraphCharacteristics\n", name_file);
    displayGraphCharacteristics(G);
    printf("GLV: NV = %ld\t NE = %ld\t numColor = %d \n", NV, NE, numColors);
}
void GLV::InitByOhterG(graphNew* G_orig) {
    assert(G_orig);
    CleanCurrentG();
    G = CloneG(G_orig);
    SyncWithG();
    InitM();
    printf("GLV: NV = %ld\t NE = %ld\t numColor = %d \n", NV, NE, numColors);
}
void GLV::SetByOhterG(graphNew* G_src) {
    assert(G_src);
    CleanCurrentG();
    G = G_src;
    SyncWithG();
    InitM();
    printf("GLV: NV = %ld\t NE = %ld\t numColor = %d \n", NV, NE, numColors);
}
void GLV::RstNVlByM() {
    NVl = 0;
    for (int i = 0; i < NV; i++)
        if (M[i] >= 0) NVl++;
}
void GLV::RstNVElg() {
    NElg = 0;
    long* vtxPtr = G->edgeListPtrs;
    edge* vtxInd = G->edgeList;
    for (long v = 0; v < NV; v++) {
        long adj1 = vtxPtr[v];
        long adj2 = vtxPtr[v + 1];
        if (M[v] < 0) continue;
        for (long k = adj1; k < adj2; k++) {
            edge e = vtxInd[k];
            if (M[e.tail] < 0) NElg++;
        }
    } // for v
}
void GLV::SetByOhterG(graphNew* G_src, long* M_src) {
    assert(G_src);
    CleanCurrentG();
    G = G_src;
    SyncWithG();
    if (M) free(M);
    M = M_src;
    RstNVlByM();
    RstNVElg();
    printf("GLV: NV = %ld\t NE = %ld\t numColor = %d \n", NV, NE, numColors);
}

void GLV::SetM(long* M_src) {
    assert(M_src);
    assert(G);
    NVl = 0;
    for (int i = 0; i < NV; i++) {
        M[i] = M_src[i];
        if (M[i] >= 0) NVl++;
    }
    RstNVElg();
}
void GLV::SetM() {
    assert(G);
    for (int i = 0; i < NV; i++) M[i] = i;
    NVl = NV;
}
void GLV::InitM(long* M_src) {
    assert(M_src);
    assert(G);
    NV = G->numVertices;
    NE = G->numEdges;
    if (M) {
        printf("GLV: InitM: M is not empty and will be free and re-allocated.\n");
        free(M);
    }
    M = (long*)malloc(NV * sizeof(long));
    SetM(M_src);
}
void GLV::InitM() {
    assert(G);
    NV = G->numVertices;
    NE = G->numEdges;
    if (M) {
        printf("GLV: InitM: M is not empty and will be free and re-allocated.\n");
        free(M);
    }
    M = (long*)malloc(NV * sizeof(long));
    SetM();
}
void GLV::SetC(long* C_src) {
    assert(C_src);
    assert(G);
    NV = G->numVertices;
    NE = G->numEdges;
    for (int i = 0; i < NV; i++) C[i] = C_src[i];
}
void GLV::SetC() {
    assert(G);
    NV = G->numVertices;
    NE = G->numEdges;
    for (int i = 0; i < NV; i++) C[i] = i;
    NC = NV;
}
void GLV::InitC(long* C_src) {
    assert(C_src);
    assert(G);
    NV = G->numVertices;
    NE = G->numEdges;
    if (C) {
        printf("GLV: InitC: C is not empty and will be free and re-allocated.\n");
        free(C);
    }
    C = (long*)malloc(NV * sizeof(long));
    SetC(C_src);
}
void GLV::InitC() {
    assert(G);
    NV = G->numVertices;
    NE = G->numEdges;
    if (C) {
        printf("GLV: InitC: C is not empty and will be free and re-allocated.\n");
        free(C);
    }
    C = (long*)malloc(NV * sizeof(long));
    SetC();
}
void GLV::ResetColor() {
    double totTimeColoring;
    numColors = Phaseloop_UsingFPGA_InitColorBuff(G, colors, numThreads, totTimeColoring);
}
void GLV::ResetC() {
    this->com_list.clear();
    InitC(); /*
     FeatureLV f1(this);
     f1.PrintFeature();
     com_list.push_back(f1);*/
}
void GLV::SyncWithG() {
    double totTimeColoring;
    assert(G);
    if (NV < G->numVertices) {
        if (C) free(C);
        C = (long*)malloc(G->numVertices * sizeof(long));
        if (colors) free(colors);
        colors = (int*)malloc(G->numVertices * sizeof(int));
    }
    NV = G->numVertices;
    NE = G->numEdges;
    ResetColor();
    ResetC();
}
void GLV::InitG(graphNew* G_src) {
    assert(G_src);
    if (G) CleanCurrentG();
    G = CloneG(G_src);
    NV = G->numVertices;
    NE = G->numEdges;
}
void GLV::SetName(char* nm) {
    strcpy(name, nm);
};
void GLV::InitColor() {
    // double totTimeColoring;
    assert(G);
    NV = G->numVertices;
    NE = G->numEdges;
    if (colors) free(colors);
    colors = (int*)malloc(G->numVertices * sizeof(int));
    ResetColor();
}
void GLV::print() {
    printSimple();
    assert(G);
    assert(C);
    assert(M);
    printG(G, C, M);
}
void GLV::printSimple() {
    // list<FeatureLV>::iterator iter = com_list.back();
    // list<FeatureLV>::iterator iter = com_list[com_list.size() - 1];
    double Q = com_list.back().Q;
    long NC = com_list.back().NC;
    if (NC == NV)
        printf("| GLV ID: %-2d| NC/NV: \033[1;37;40m%8d\033[0m/", ID, NC, NVl);
    else
        printf("| GLV ID: %-2d| NC/NV: \033[1;31;40m%8d\033[0m/", ID, NC, NVl);
    if (NV < (1)) {
        if (NV == NVl)
            printf(" \033[1;37;40m%-3d\033[0m(%-3d/%2d\%)", NV, (NV - NVl), (int)(100 * (float)(NV - NVl) / (float)NV));
        else
            printf(" \033[1;37;40m%-3d\033[0m(%-3d/%2d\%)", NV, (NV - NVl), (int)(100 * (float)(NV - NVl) / (float)NV));
    } else if (NV < (1000000)) {
        if (NV == NVl)
            printf(" \033[1;37;40m%-6d\033[0m(%-5d/%2d\%)", NV, (NV - NVl), (int)(100 * (float)(NV - NVl) / (float)NV));
        else
            printf(" \033[1;37;40m%-6d\033[0m(%-5d/%2d\%)", NV, (NV - NVl), (int)(100 * (float)(NV - NVl) / (float)NV));
    } else {
        if (NV == NVl)
            printf(" \033[1;37;40m%-6d\033[0m(%-8d/%2d\%)", NV, (NV - NVl), (int)(100 * (float)(NV - NVl) / (float)NV));
        else
            printf(" \033[1;37;40m%-6d\033[0m(%-8d/%2d\%)", NV, (NV - NVl), (int)(100 * (float)(NV - NVl) / (float)NV));
    }
    if (NE < (1000)) {
        if (NElg == 0)
            printf(" NE: %9d(%-9d/%2d\%)| Colors:%-6d ", NE, NElg, (int)(100 * (float)NElg / (float)NE), numColors);
        else
            printf(" NE: %9d(%-9d/%2d\%)| Colors:%-6d ", NE, NElg, (int)(100 * (float)NElg / (float)NE), numColors);
    } else if (NE < (1000000)) {
        if (NElg == 0)
            printf(" NE: %9d(%-9d/%2d\%)| Colors:%-6d ", NE, NElg, (int)(100 * (float)NElg / (float)NE), numColors);
        else
            printf(" NE: %9d(%-9d/%2d\%)| Colors:%-6d ", NE, NElg, (int)(100 * (float)NElg / (float)NE), numColors);
    } else {
        if (NElg == 0)
            printf(" NE: %9d(%-9d/% 2d\%)| Colors:%-6d ", NE, NElg, (int)(100 * (float)NElg / (float)NE), numColors);
        else
            printf(" NE: %9d(%-9d/%2d\%)| Colors:%-6d ", NE, NElg, (int)(100 * (float)NElg / (float)NE), numColors);
    }
    if (Q > 0)
        printf(" Q: \033[1;32;40m%1.6f\033[0m  ", Q);
    else
        printf(" Q:\033[1;37;40m%2.6f\033[0m  ", Q);
    printf("| name: %s \n", name);
}
void GLV::PushFeature(int ph, int iter, double time, bool FPGA) {
    FeatureLV f1(this);
    f1.No_phase = ph;
    f1.Num_iter = iter;
    f1.time = time;
    f1.isFPGA = FPGA;
    this->com_list.push_back(f1);
    this->NC = f1.NC;
}
void GLV::printFeature() {
    list<FeatureLV>::iterator iter = com_list.begin();
    while (iter != com_list.end()) {
        (*iter).PrintFeature();
        iter++;
    }
}
void GLV::SetName_par(int ID_par, int ID_src, long start, long end, int th) {
    char nm[256];
    sprintf(nm, "ID:%d_ParID:%d_%d_%d_th%d", ID_par, ID_src, start, end, th);
    this->SetName(nm);
}
void GLV::SetName_lv(int ID_par, int ID_src) {
    char nm[256];
    sprintf(nm, "ID:%d_lv(ID:%d)", ID_par, ID_src);
    this->SetName(nm);
}
void GLV::SetName_ParLvMrg(int num_par, int ID_src) {
    char nm[256];
    sprintf(nm, "ID:%d_Mrg(lv(Par%d(ID:%d)))", ID, num_par, ID_src);
    this->SetName(nm);
}
void GLV::SetName_loadg(int ID_curr, char* path) {
    char nm[256];
    sprintf(nm, "ID:%d_%s", ID_curr, path);
    this->SetName(nm);
}
void GLV::SetName_cat(int ID_src1, int ID_src2) {
    char nm[256];
    sprintf(nm, "ID_%d_cat(ID_%d_ID%d)))", ID, ID_src1, ID_src2);
    this->SetName(nm);
}
////////////////////////////////////////////////////////////////////////////
SttGPar::SttGPar(long s, long e) {
    assert(e > s);
    start = s;
    end = e;
    num_v = num_e = num_e_dir = num_e_ll_dir = 0;
    num_v_l = num_v_g = 0;
    num_e_ll = num_e_lg = num_e_gl = num_e_gg = 0;
}
SttGPar::SttGPar() {
    num_v = num_e = num_e_dir = num_e_ll_dir = 0;
    num_v_l = num_v_g = 0;
    num_e_ll = num_e_lg = num_e_gl = num_e_gg = 0;
}
void SttGPar::PrintStt() {
    printf("**SttGPar::PrintStt BEGIN**\n");
    printf("From %ld to %ld \n", start, end);
    num_v_l = end - start;
    printf("Total V : %ld\t Total  Vl : %ld\t Total Vg: %ld\t Vl\/V=%2.2f\%\n", num_v, num_v_l, num_v_g,
           (float)num_v_l / (float)num_v * 100.0);
    assert(num_e_lg == num_e - num_e_ll);
    printf("Total 2E: %ld\t Total  ll : %ld\t Total lg: %ld\t ll\/E=%2.2f\%\n", num_e, num_e_ll, num_e_lg,
           (float)num_e_ll / (float)num_e * 100.0);
    printf("Total|E|: %ld\t Total |ll|: %ld\t Total lg: %ld\t |ll|\/|E|=%2.2f\%\n", num_e_dir, num_e_ll_dir, num_e_lg,
           (float)num_e_ll_dir / (float)num_e_dir * 100.0);
    printf("**SttGPar::PrintStt END**\n");
}

bool SttGPar::InRange(long v) {
    return v >= start && v < end;
}
void SttGPar::AddEdge(edge* edges, long head, long tail, double weight, long* M_g) {
    long head_m = head - start;
    long off = end - start;
    long tail_m;
    map<long, long>::iterator itr;
    num_e++;
    // num_e
    if (InRange(tail)) {
        num_e_ll++;
        if (head <= tail) {
            tail_m = tail - start;
            // printf("NODIR(%ld)\t: ll:<%ld %ld> -> <%ld %ld> \t= %ld - %ld \n", num_e_dir, head, tail, head_m, tail_m,
            // tail, start);
            edges[num_e_dir].head = head_m;
            edges[num_e_dir].tail = tail_m;
            edges[num_e_dir].weight = weight;
            num_e_ll_dir++;
            num_e_dir++;
        }
    } else {
        itr = map_v_g.find(tail);
        if (itr == map_v_g.end()) {
            tail_m = num_v_g + off;
            M_g[tail_m] = -tail - 1; // using negtive to indicate it ghost
            map_v_g[tail] = num_v_g++;
            // printf("NODIR(%ld)\t: lg:<%ld %ld> -> <%ld %ld>  \t= %ld + %ld \n", num_e_dir, head, tail, head_m,
            // tail_m, num_v_g, off);
        } else {
            tail_m = itr->second + off;
            // printf("NODIR(%ld)\t: lg:<%ld %ld> -> <%ld %ld>  \t= %ld + %ld \n", num_e_dir, head, tail, head_m,
            // tail_m, itr->second, off);
        }
        edges[num_e_dir].head = head_m;
        edges[num_e_dir].tail = tail_m;
        edges[num_e_dir].weight = weight;
        num_e_lg++;
        num_e_dir++;
    }
    num_v_l = end - start;
    num_v = num_v_l + num_v_g;
}

long SttGPar::findAt(VGMinDgr& gMinDgr, long tail, long dgr, long num_vg, int th_maxGhost) {
    long index = 0;
    long low = 0, high;

    if (th_maxGhost == 1) {
        if (dgr < gMinDgr.dgrs[0] || (dgr == gMinDgr.dgrs[0] && tail < gMinDgr.tail[0]))
            return 0;
        else
            return -1;
    } else if (th_maxGhost > 1) { // th_maxGhost > 1 && num_vg < th_maxGhost
        high = num_vg < th_maxGhost ? num_vg - 1 : th_maxGhost - 1;
        if (dgr < gMinDgr.dgrs[0] || (dgr == gMinDgr.dgrs[0] && tail < gMinDgr.tail[0]))
            return 0;
        else {
            while (low <= high) {
                index = (low + high) / 2;
                if (dgr == gMinDgr.dgrs[index] && tail == gMinDgr.tail[index])
                    return -1;
                else if (dgr < gMinDgr.dgrs[index] || (dgr == gMinDgr.dgrs[index] && tail < gMinDgr.tail[index]))
                    high = index - 1;
                else if (dgr > gMinDgr.dgrs[index] || (dgr == gMinDgr.dgrs[index] && tail > gMinDgr.tail[index]))
                    low = index + 1;
            }
            return low;
        }
    } else
        return -1;
}

void SttGPar::EdgePruning(edge* edges,
                          long head,
                          long tail,
                          double weight,
                          long* M_g,
                          VGMinDgr& gMinDgr,
                          long& num_vg,
                          long& e_dgr,
                          int th_maxGhost) {
    long head_m = head - start;
    long tail_m;

    // num_e
    if (InRange(tail)) {
        num_e++;
        num_e_ll++;
        if (head <= tail) {
            tail_m = tail - start;
            // printf("NODIR(%ld)\t: ll:<%ld %ld> -> <%ld %ld> \t= %ld - %ld \n", num_e_dir, head, tail, head_m, tail_m,
            // tail, start);
            edges[num_e_dir].head = head_m;
            edges[num_e_dir].tail = tail_m;
            edges[num_e_dir].weight = weight;
            num_e_ll_dir++;
            num_e_dir++;
        }
    } else {
        // printf("tail=%ld dgr= %ld \n", tail, e_dgr);
        if (num_vg == 0) {
            gMinDgr.tail[0] = tail;
            gMinDgr.dgrs[0] = e_dgr;
            gMinDgr.wght[0] = weight;
        } else {
            long at = findAt(gMinDgr, tail, e_dgr, num_vg, th_maxGhost);
            if (at >= 0 && at < th_maxGhost) {
                long where = num_vg < th_maxGhost ? num_vg : (th_maxGhost - 1);
                for (int i = where; i > at; i--) {
                    gMinDgr.tail[i] = gMinDgr.tail[i - 1];
                    gMinDgr.dgrs[i] = gMinDgr.dgrs[i - 1];
                    gMinDgr.wght[i] = gMinDgr.wght[i - 1];
                }
                gMinDgr.tail[at] = tail;
                gMinDgr.dgrs[at] = e_dgr;
                gMinDgr.wght[at] = weight;
                //              printf("insert: at= %ld \n", at);
            }
        }
        num_vg++;
    }
}

void SttGPar::CountVPruning(graphNew* G, long st, long ed, int th_maxGhost) {
    assert(G);
    assert(ed > st);

    start = st;
    end = ed;
    long NV = G->numVertices;
    long* vtxPtr = G->edgeListPtrs;
    edge* vtxInd = G->edgeList;
    assert(end <= NV);
    long ne = vtxPtr[end] - vtxPtr[start];
    edge* elist = (edge*)malloc(sizeof(edge) * (ne));
    long* M_g = (long*)malloc(sizeof(long) * (NV));

    // long off = end - start;

    long off = end - start;
    for (int i = 0; i < NV; i++) {
        M_g[i] = i < off ? i + start : -2;
    }
    for (long v = start; v < end; v++) {
        map<long, long>::iterator itr;
        VGMinDgr gMinDgr;
        long num_vg = 0;
        long e_dgr = 0;
        long head_m, tail_m;
        long adj1 = vtxPtr[v];
        long adj2 = vtxPtr[v + 1];
        int degree = adj2 - adj1;
        for (int d = 0; d < degree; d++) {
            long e = vtxInd[adj1 + d].tail;
            double w = vtxInd[adj1 + d].weight;
            e_dgr = vtxPtr[e + 1] - vtxPtr[e];
            head_m = v - start;
            EdgePruning(elist, v, e, w, M_g, gMinDgr, num_vg, e_dgr, th_maxGhost);
        } // for
        long smallest = num_vg < th_maxGhost ? num_vg : th_maxGhost;
        for (int i = 0; i < smallest; i++) {
            itr = map_v_g.find(gMinDgr.tail[i]);
            if (itr == map_v_g.end()) {
                tail_m = num_v_g + off;
                M_g[tail_m] = -gMinDgr.tail[i] - 1;
                map_v_g[gMinDgr.tail[i]] = num_v_g++;
            } else {
                tail_m = itr->second + off;
            }
            elist[num_e_dir].head = head_m;
            elist[num_e_dir].tail = tail_m;
            elist[num_e_dir].weight = gMinDgr.wght[i];
            num_e_lg++;
            num_e_dir++;
            num_e++;
            printf("vertex= %ld\t nGhost= %ld\t sGhost= %ld\t  degree= %ld\t\n", v, num_vg, gMinDgr.tail[i],
                   gMinDgr.dgrs[i]);
        }
    }

    num_v_l = end - start;
    num_v = num_v_l + num_v_g;
    graphNew* Gnew = (graphNew*)malloc(sizeof(graphNew));
    GetGFromEdge(Gnew, elist, num_v, num_e_dir);
    // printG(Gnew);
    printG(Gnew, M_g);
    FreeG(Gnew);
    free(elist);
    free(M_g);
}

void SttGPar::CountV(graphNew* G, long st, long ed, edge* elist, long* M_g) {
    assert(G);
    assert(ed > st);
    start = st;
    end = ed;
    long NV = G->numVertices;
    long* vtxPtr = G->edgeListPtrs;
    edge* vtxInd = G->edgeList;
    assert(end <= NV);
    long ne = vtxPtr[end] - vtxPtr[start];
    elist = (edge*)malloc(sizeof(edge) * (ne));
    M_g = (long*)malloc(sizeof(long) * (NV));

    // long off = end - start;

    long off = end - start;
    for (int i = 0; i < NV; i++) {
        M_g[i] = i < off ? i + start : -2;
    }
    for (long v = start; v < end; v++) {
        long adj1 = vtxPtr[v];
        long adj2 = vtxPtr[v + 1];
        int degree = adj2 - adj1;
        for (int d = 0; d < degree; d++) {
            long e = vtxInd[adj1 + d].tail;
            double w = vtxInd[adj1 + d].weight;
            AddEdge(elist, v, e, w, M_g);
        } // for
    }
    graphNew* Gnew = (graphNew*)malloc(sizeof(graphNew));
    GetGFromEdge(Gnew, elist, num_v, num_e_dir);
    // printG(Gnew);
    printG(Gnew, M_g);
    FreeG(Gnew);
    free(elist);
    free(M_g);
}

void SttGPar::CountV(graphNew* G, long st, long ed) {
    assert(G);
    assert(ed > st);
    start = st;
    end = ed;
    long NV = G->numVertices;
    long* vtxPtr = G->edgeListPtrs;
    edge* vtxInd = G->edgeList;
    assert(end <= NV);
    long ne = vtxPtr[end] - vtxPtr[start];
    edge* elist = (edge*)malloc(sizeof(edge) * (ne));
    long* M_g = (long*)malloc(sizeof(long) * (NV));

    // long off = end - start;

    long off = end - start;
    for (int i = 0; i < NV; i++) {
        M_g[i] = i < off ? i + start : -2;
    }
    for (long v = start; v < end; v++) {
        long adj1 = vtxPtr[v];
        long adj2 = vtxPtr[v + 1];
        int degree = adj2 - adj1;
        for (int d = 0; d < degree; d++) {
            long e = vtxInd[adj1 + d].tail;
            double w = vtxInd[adj1 + d].weight;
            AddEdge(elist, v, e, w, M_g);
        } // for
    }
    graphNew* Gnew = (graphNew*)malloc(sizeof(graphNew));
    GetGFromEdge(Gnew, elist, num_v, num_e_dir);
    // printG(Gnew);
    printG(Gnew, M_g);
    FreeG(Gnew);
    free(elist);
    free(M_g);
}
GLV* CloneGlv(GLV* glv_src, int& id_glv) {
    assert(glv_src);
    GLV* glv = new GLV(id_glv);
    glv->InitG(glv_src->G);
    glv->InitC(glv_src->C);
    glv->InitM(glv_src->M);
    glv->InitColor();
    return glv;
}
GLV* GLV::CloneSelf(int& id_glv) {
    GLV* glv = new GLV(id_glv);
    glv->InitG(G);
    glv->InitC(C);
    glv->InitM(M);
    glv->InitColor();
    return glv;
}
GLV* SttGPar::ParNewGlv(graphNew* G_src, long st, long ed, int& id_glv) {
    assert(G_src);
    assert(ed > st);
    start = st;
    end = ed;
    long NV = G_src->numVertices;
    long* vtxPtr = G_src->edgeListPtrs;
    edge* vtxInd = G_src->edgeList;
    assert(end <= NV);

    long ne = vtxPtr[end] - vtxPtr[start];
    edge* elist = (edge*)malloc(sizeof(edge) * (ne));
    long* M_v = (long*)malloc(sizeof(long) * (NV)); // address by v

    long off = end - start;
    for (int i = 0; i < NV; i++) {
        M_v[i] = i < off ? i + start : -2;
    }

    for (long v = start; v < end; v++) {
        long adj1 = vtxPtr[v];
        long adj2 = vtxPtr[v + 1];
        int degree = adj2 - adj1;
        for (int d = 0; d < degree; d++) {
            long e = vtxInd[adj1 + d].tail;
            double w = vtxInd[adj1 + d].weight;
            AddEdge(elist, v, e, w, M_v);
        } // for
    }
    graphNew* Gnew = (graphNew*)malloc(sizeof(graphNew));
    GLV* glv = new GLV(id_glv);

    GetGFromEdge(Gnew, elist, num_v, num_e_dir);
    glv->SetByOhterG(Gnew);
    glv->SetM(M_v);
    // printG(Gnew);
    // FreeG(Gnew);
    free(elist);
    free(M_v);
    glv->NVl = ed - st;
    glv->RstNVElg();
    return glv;
}
GLV* SttGPar::ParNewGlv_Prun(graphNew* G, long st, long ed, int& id_glv, int th_maxGhost) {
    start = st;
    end = ed;
    long NV = G->numVertices;
    long* vtxPtr = G->edgeListPtrs;
    edge* vtxInd = G->edgeList;
    assert(end <= NV);
    long ne = vtxPtr[end] - vtxPtr[start];
    edge* elist = (edge*)malloc(sizeof(edge) * (ne));
    long* M_v = (long*)malloc(sizeof(long) * (NV));

    // long off = end - start;

    long off = end - start;
    for (int i = 0; i < NV; i++) {
        M_v[i] = i < off ? i + start : -2;
    }
    for (long v = start; v < end; v++) {
        map<long, long>::iterator itr;
        VGMinDgr gMinDgr;
        long num_vg = 0;
        long e_dgr = 0;
        long head_m, tail_m;
        long adj1 = vtxPtr[v];
        long adj2 = vtxPtr[v + 1];
        int degree = adj2 - adj1;
        for (int d = 0; d < degree; d++) {
            long e = vtxInd[adj1 + d].tail;
            double w = vtxInd[adj1 + d].weight;
            e_dgr = vtxPtr[e + 1] - vtxPtr[e];
            head_m = v - start;
            EdgePruning(elist, v, e, w, M_v, gMinDgr, num_vg, e_dgr, th_maxGhost);
        } // for
        long smallest = num_vg < th_maxGhost ? num_vg : th_maxGhost;
        for (int i = 0; i < smallest; i++) {
            itr = map_v_g.find(gMinDgr.tail[i]);
            if (itr == map_v_g.end()) {
                tail_m = num_v_g + off;
                M_v[tail_m] = -gMinDgr.tail[i] - 1;
                map_v_g[gMinDgr.tail[i]] = num_v_g++;
            } else {
                tail_m = itr->second + off;
            }
            elist[num_e_dir].head = head_m;
            elist[num_e_dir].tail = tail_m;
            elist[num_e_dir].weight = gMinDgr.wght[i];
            num_e_lg++;
            num_e_dir++;
            num_e++;
            // printf("vertex= %ld\t nGhost= %ld\t sGhost= %ld\t  degree= %ld\t\n", v, num_vg, gMinDgr.tail[i],
            // gMinDgr.dgrs[i]);
        }
    }
    num_v_l = end - start;
    num_v = num_v_l + num_v_g;
    graphNew* Gnew = (graphNew*)malloc(sizeof(graphNew));
    GLV* glv = new GLV(id_glv);

    GetGFromEdge(Gnew, elist, num_v, num_e_dir);
    glv->SetByOhterG(Gnew);
    glv->SetM(M_v);
    // printG(Gnew);
    // FreeG(Gnew);
    free(elist);
    free(M_v);
    return glv;
}
void SttGPar::CountV(graphNew* G, edge* elist, long* M_g) {
    long NV = G->numVertices;
    return CountV(G, 0, NV, elist, M_g);
}
void SttGPar::CountV(graphNew* G) {
    long NV = G->numVertices;
    return CountV(G, 0, NV);
}
void GetGFromEdge(graphNew* G, edge* edgeListTmp, long num_v, long num_e_dir) {
    // Parse the first line:

    long NV = num_v, ED = num_e_dir * 2, NE = num_e_dir;

    // printf("Done reading from edges.\n");
    printf("|V|= %ld, |E|= %ld \n", NV, NE);

    // Remove duplicate entries:

    // Allocate for Edge Pointer and keep track of degree for each vertex
    long* edgeListPtr = (long*)malloc((NV + 1) * sizeof(long));
#pragma omp parallel for
    for (long i = 0; i <= NV; i++) edgeListPtr[i] = 0; // For first touch purposes

#pragma omp parallel for
    for (long i = 0; i < NE; i++) {
        __sync_fetch_and_add(&edgeListPtr[edgeListTmp[i].head + 1], 1); // Plus one to take care of the zeroth location
        __sync_fetch_and_add(&edgeListPtr[edgeListTmp[i].tail + 1], 1);
    }
    double time1, time2;
    //////Build the EdgeListPtr Array: Cumulative addition
    time1 = omp_get_wtime();
    for (long i = 0; i < NV; i++) {
        edgeListPtr[i + 1] += edgeListPtr[i]; // Prefix Sum:
    }
    // The last element of Cumulative will hold the total number of characters
    time2 = omp_get_wtime();
    // printf("Done cumulative addition for edgeListPtrs:  %9.6lf sec.\n", time2 - time1);
    printf("Sanity Check: 2|E| = %ld, edgeListPtr[NV]= %ld\n", NE * 2, edgeListPtr[NV]);

    //  printf("About to allocate memory for graph data structures\n");
    time1 = omp_get_wtime();
    edge* edgeList = (edge*)malloc((2 * NE) * sizeof(edge)); // Every edge stored twice
    assert(edgeList != 0);
    // Keep track of how many edges have been added for a vertex:
    long* added = (long*)malloc(NV * sizeof(long));
    assert(added != 0);
#pragma omp parallel for
    for (long i = 0; i < NV; i++) added[i] = 0;
    time2 = omp_get_wtime();
    // printf("Time for allocating memory for edgeList = %lf\n", time2 - time1);

    time1 = omp_get_wtime();

    // printf("About to build edgeList...\n");
    // Build the edgeList from edgeListTmp:
    //#pragma omp parallel for
    for (long i = 0; i < NE; i++) {
        long head = edgeListTmp[i].head;
        long tail = edgeListTmp[i].tail;
        double weight = edgeListTmp[i].weight;

        long Where = edgeListPtr[head] + __sync_fetch_and_add(&added[head], 1);
        edgeList[Where].head = head;
        edgeList[Where].tail = tail;
        edgeList[Where].weight = weight;
        // added[head]++;
        // Now add the counter-edge:
        Where = edgeListPtr[tail] + __sync_fetch_and_add(&added[tail], 1);
        edgeList[Where].head = tail;
        edgeList[Where].tail = head;
        edgeList[Where].weight = weight;
        // added[tail]++;
    }
    time2 = omp_get_wtime();
    //// printf("Time for building edgeList = %lf\n", time2 - time1);

    G->sVertices = NV;
    G->numVertices = NV;
    G->numEdges = NE;
    G->edgeListPtrs = edgeListPtr;
    G->edgeList = edgeList;

    // free(edgeListTmp);
    free(added);
}

long GetGFromEdge_selfloop(graphNew* G, edge* edgeListTmp, long num_v, long num_e_dir) {
    long NV = num_v, ED = num_e_dir * 2, NE = num_e_dir;
    // Remove duplicate entries:
    /* long NewEdges = removeEdges(NV, NE, edgeListTmp);
     if (NewEdges < NE) {
       printf("GetGFromEdge_selfloop: Number of duplicate entries detected: %ld\n", NE-NewEdges);
       NE = NewEdges; //Only look at clean edges
     }*/
    printf("|V|= %ld, |E|= %ld \n", NV, NE);

    long* edgeListPtr = (long*)malloc((NV + 1) * sizeof(long));
#pragma omp parallel for
    for (long i = 0; i <= NV; i++) edgeListPtr[i] = 0; // For first touch purposes

#pragma omp parallel for
    for (long i = 0; i < NE; i++) {
        __sync_fetch_and_add(&edgeListPtr[edgeListTmp[i].head + 1], 1); // Plus one to take care of the zeroth location
        if (edgeListTmp[i].head != edgeListTmp[i].tail) __sync_fetch_and_add(&edgeListPtr[edgeListTmp[i].tail + 1], 1);
    }
    double time1, time2;
    //////Build the EdgeListPtr Array: Cumulative addition
    time1 = omp_get_wtime();
    for (long i = 0; i < NV; i++) {
        edgeListPtr[i + 1] += edgeListPtr[i]; // Prefix Sum:
    }
    // The last element of Cumulative will hold the total number of characters
    time2 = omp_get_wtime();
    // printf("Done cumulative addition for edgeListPtrs:  %9.6lf sec.\n", time2 - time1);
    printf("Sanity Check: 2|E| = %ld, edgeListPtr[NV]= %ld  GetGFromEdge_selfloop\n", NE * 2, edgeListPtr[NV]);

    //  printf("About to allocate memory for graph data structures\n");
    time1 = omp_get_wtime();
    edge* edgeList = (edge*)malloc((2 * NE) * sizeof(edge)); // Every edge stored twice
    assert(edgeList != 0);
    // Keep track of how many edges have been added for a vertex:
    long* added = (long*)malloc(NV * sizeof(long));
    assert(added != 0);
#pragma omp parallel for
    for (long i = 0; i < NV; i++) added[i] = 0;
    time2 = omp_get_wtime();
    // printf("Time for allocating memory for edgeList = %lf\n", time2 - time1);

    time1 = omp_get_wtime();

    // printf("About to build edgeList...\n");
    // Build the edgeList from edgeListTmp:
    //#pragma omp parallel for
    for (long i = 0; i < NE; i++) {
        long head = edgeListTmp[i].head;
        long tail = edgeListTmp[i].tail;
        double weight = edgeListTmp[i].weight;

        long Where = edgeListPtr[head] + __sync_fetch_and_add(&added[head], 1);
        edgeList[Where].head = head;
        edgeList[Where].tail = tail;
        edgeList[Where].weight = weight;
        // added[head]++;
        // Now add the counter-edge:
        // printf("GetGFromEdge_selfloop e=%-6d head=%-6d tail=%-6d w=%-4.0f where(%d)=%d + added[head](%d) \n",i, head,
        // tail, weight, Where, edgeListPtr[head], added[head]);
        if (head != tail) {
            Where = edgeListPtr[tail] + __sync_fetch_and_add(&added[tail], 1);
            edgeList[Where].head = tail;
            edgeList[Where].tail = head;
            edgeList[Where].weight = weight;
        }
        // added[tail]++;
    }
    time2 = omp_get_wtime();
    //// printf("Time for building edgeList = %lf\n", time2 - time1);

    G->sVertices = NV;
    G->numVertices = NV;
    G->numEdges = NE;
    G->edgeListPtrs = edgeListPtr;
    G->edgeList = edgeList;

    // free(edgeListTmp);
    free(added);
    return NE;
}
//////////////////////////////////////////////////////////////////
double FeatureLV::ComputeQ(GLV* glv) {
    assert(glv->G);
    NV = glv->G->numVertices;
    NE = glv->G->numEdges;
    long* vtxPtr = glv->G->edgeListPtrs;
    edge* vtxInd = glv->G->edgeList;
    long* tot_m = (long*)malloc(sizeof(long) * NV);
    for (int v = 0; v < NV; v++) tot_m[v] = 0;
    m = 0;
    for (int v = 0; v < NV; v++) {
        long cid = glv->C[v];
        long adj1 = vtxPtr[v];
        long adj2 = vtxPtr[v + 1];
        int degree = adj2 - adj1;
        for (int d = 0; d < degree; d++) {
            long e = vtxInd[adj1 + d].tail;
            double w = vtxInd[adj1 + d].weight;
            long cide = glv->C[e];
            m += w;
            tot_m[cide] += w;
            if (cide == cid) totalIn += w;
        }
    }
    NC = 0;
    for (int v = 0; v < NV; v++) {
        totalTot += tot_m[v] * tot_m[v];
        if (tot_m[v]) NC++;
    }
    Q = totalIn / m - totalTot / (m * m);
    free(tot_m);
    return Q;
}
double FeatureLV::ComputeQ2(GLV* glv) {
    assert(glv->G);
    NV = glv->G->numVertices;
    NE = glv->G->numEdges;
    long* vtxPtr = glv->G->edgeListPtrs;
    edge* vtxInd = glv->G->edgeList;
    long* tot_m = (long*)malloc(sizeof(long) * NV);
    for (int v = 0; v < NV; v++) tot_m[v] = 0;
    m = 0;
    for (int e = 0; e < NE; e++) {
        long head = vtxInd[e].head;
        long tail = vtxInd[e].tail;
        double w = vtxInd[e].weight;
        long cid = glv->C[head];
        long cide = glv->C[tail];
        m += w;
        tot_m[cide] += w;
        if (cide == cid) totalIn += w;
    }
    NC = 0;
    for (int v = 0; v < NV; v++) {
        totalTot += tot_m[v] * tot_m[v];
        if (tot_m[v]) NC++;
    }
    Q = totalIn / m - totalTot / (m * m);
    free(tot_m);
    return Q;
}
void FeatureLV::PrintFeature() {
    printf("NC=%-8d  NV=%-8d  NE=%-8d  ", NC, NV, NE);
    printf("Q=%-2.6f   m=%-8.1f    totalTot=%-14.1f  totalIn=%-8.1f ", Q, m, totalTot, totalIn);
    printf("No_phase=%-2d   Num_iter=%-2d    time=%-8.1f  %s \n", No_phase, Num_iter, time,
           isFPGA == true ? "FPGA" : "CPU");
}
FeatureLV::FeatureLV(GLV* glv) {
    init();
    ComputeQ(glv);
}
void FeatureLV::init() {
    NV = NE = 0;
    totalTot = totalIn = m = 0;
    Q = -1;
    NC = 0;
    No_phase = Num_iter = 0;
    time = 0;
    isFPGA = true;
}
FeatureLV::FeatureLV() {
    init();
} // FeatureLV()
/*
//louvainPhase.cpp///////////////////////////////////////////////////
int Phaseloop_UsingFPGA_InitColorBuff(
                graphNew *G,
                int *colors,
                int numThreads,
                double &totTimeColoring){
#pragma omp parallel for
    for (long i = 0; i < G->numVertices; i++) {
      colors[i] = -1;
    }
    double tmpTime;
    int numColors = algoDistanceOneVertexColoringOpt(G, colors, numThreads, &tmpTime) + 1;
    totTimeColoring += tmpTime;
    return numColors;
}*/
