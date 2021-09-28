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
#include "ParLV.h"
#include "partitionLouvain.hpp"
#include "louvainPhase.h"
#include "ctrlLV.h"
#include <thread>

/*
void ParLV::CleanList(GLV* glv_curr, GLV* glv_temp){
        list<GLV*>::iterator iter;
        iter = par_list.begin();
        while(iter != par_list.end())
        {
                printf("\033[1;37;40mINFO\033[0m: Deleting ParLV ID:%d name:%s\n", (*iter)->ID, (*iter)->name);
                if(*iter==glv_curr)
                        glv_curr=NULL;
                if(*iter==glv_temp)
                        glv_temp=NULL;
                delete(*iter);
                iter++;
        }
}*/
void ParLV::Init(int mode) {
    st_Partitioned = false;
    st_ParLved = false;
    st_PreMerged = false;
    st_Merged = false;
    st_FinalLved = false;
    //
    st_Merged_ll = false;
    st_Merged_gh = false;
    isMergeGhost = false;
    isOnlyGL = false;
    isPrun = true;
    th_prun = 1;
    plv_src = NULL;
    plv_merged = NULL;
    plv_final = NULL;
    num_par = NV = NVl = NE = NElg = NEll = NEgl = NEgg = NEself = NV_gh = 0;
    elist = NULL;
    M_v = NULL;
    NE_list_all = 0;
    NE_list_ll = 0;
    NE_list_gl = 0;
    NE_list_gg = 0;
    NV_list_all = 0;
    NV_list_l = 0;
    NV_list_g = 0;
    num_dev = 1;
    flowMode = mode;
}
void ParLV::Init(int mode, GLV* src, int nump, int numd) {
    Init(mode);
    plv_src = src;
    num_par = nump;
    num_dev = numd;
    st_Partitioned = true;
}
void ParLV::Init(int mode, GLV* src, int num_p, int num_d, bool isPrun, int th_prun) {
    Init(mode, src, num_p, num_d);
    this->isPrun = isPrun;
    this->th_prun = th_prun;
}
ParLV::ParLV() {
    Init(MD_FAST);
}
ParLV::~ParLV() {
    num_par = 0;
    if (elist) free(elist);
    if (M_v) free(M_v);
}
void ParLV::PrintSelf() {
    printf("PAR: %s is partitioned into %d share:\n", plv_src->name, num_par); // basic information
    for (int p = 0; p < num_par; p++) {
        printf("Share-%d    %s \t: NV=%-8d NVl=%-8d NE=%-8d NElg=%-8d \n", p, par_src[p]->name, par_src[p]->NV,
               par_src[p]->NVl, par_src[p]->NE, par_src[p]->NElg); // sub graphNew information
        if (st_ParLved == false) continue;
        printf("Share-%d Lv(%s)\t: NV=%-8d NVl=%-8d NE=%-8d NElg=%-8d \n\n", p, par_src[p]->name, par_lved[p]->NV,
               par_lved[p]->NVl, par_lved[p]->NE, par_lved[p]->NElg); // sub graphNew information
    }
    for (int p = 0; p < num_par; p++)
        if (st_Partitioned == false)
            break;
        else
            this->par_src[p]->printSimple();

    for (int p = 0; p < num_par; p++)
        if (st_ParLved == false)
            break;
        else
            this->par_lved[p]->printSimple();

    if (st_FinalLved) {
        this->plv_src->printSimple();
        this->plv_merged->printSimple();
        this->plv_final->printSimple();
        printf("NE(%d) = NEl(%d) + NE_ll(%d) + NE_lg(%d) + NE_gg(%d)\n", NE, NEll, NElg, NEgg);
        printf("NEself = %d\n", NEself);
    } else if (st_Merged) {
        this->plv_src->printSimple();
        this->plv_merged->printSimple();
        printf("NE(%d) = NEl(%d) + NE_ll(%d) + NE_lg(%d) + NE_gg(%d)\n", NE, NEll, NElg, NEgg);
        printf("NEself = %d\n", NEself);
    } else if (st_PreMerged) {
        this->plv_merged->printSimple();
        printf("NV(%d) = NVl(%d) + NV_gh(%d)\n", NV, NVl, NV_gh);
    }
}

double ParLV::UpdateTimeAll() {
    timesPar.timeAll =
        +timesPar.timePar_all + timesPar.timeLv_all + timesPar.timePre + timesPar.timeMerge + timesPar.timeFinal;
};
int ParLV::partition(GLV* glv_src, int& id_glv, int num, long th_size, int th_maxGhost) {
    assert(glv_src);
    assert(glv_src->G);
    num_par = num;
    if (num_par >= MAX_PARTITION) {
        printf("\033[1;31;40mERROR\033[0m: exe_LV_SETM wrong number of partition %d which should be small than %d!\n",
               num_par, MAX_PARTITION);
        return -1;
    }
    long vsize = glv_src->NV / num_par;
    long start = 0;
    long end = start + vsize;
    off_src[0] = 0;
    for (int i = 0; i < num_par; i++) {
        if (th_maxGhost > 0)
            par_src[i] = stt[i].ParNewGlv_Prun(glv_src->G, start, end, id_glv, th_maxGhost);
        else
            par_src[i] = stt[i].ParNewGlv(glv_src->G, start, end, id_glv);
        // par_list.push_back(pt_par[i]);
        start = end;
        end = start + vsize;
        off_src[i + 1] = start;
    }
    return 0;
}
int GetScl(long v) {
    int ret = 0;
    while (v > 0) {
        v = v >> 1;
        ret++;
    }
    return ret;
}
void ParLV::PreMerge() {
    // Do following things
    // 1) Get real NVl by accumulating NVl of louvained sub-graphNews
    // 2) Get real NV_gh and find target C or created new C
    // 3) Get real NV by using NVl + NV_gh
    // 4) Estimate NE by accumulating NE of louvained sub-graphNews
    // 5) Allocate memory for elist by estimated NE which will be used for create new merged graphNew
    // 6) Allocate memory for M_v by real NV which will be used for create new merged graphNew
    // 7) Update C of source sub graphNew
    if (st_PreMerged == true) return;
    assert(num_par > 0);
    assert(st_ParLved == true);
    off_lved[0] = off_src[0] = 0;
    NV = NVl = NE = NElg = 0;
    max_NV = max_NVl = max_NE = max_NElg = 0;
    for (int p = 0; p < num_par; p++) {
        NV += par_lved[p]->NV;
        NVl += par_lved[p]->NVl;
        NE += par_lved[p]->NE;
        NElg += par_lved[p]->NElg;
        max_NV = max_NV > par_lved[p]->NV ? max_NV : par_lved[p]->NV;
        max_NVl = max_NVl > par_lved[p]->NVl ? max_NVl : par_lved[p]->NVl;
        max_NE = max_NE > par_lved[p]->NE ? max_NE : par_lved[p]->NE;
        max_NElg = max_NElg > par_lved[p]->NElg ? max_NElg : par_lved[p]->NElg;
        off_lved[p + 1] = NVl;
        off_src[p + 1] = off_src[p] + par_src[p]->NVl;
    }
    scl_NV = GetScl(max_NV);
    scl_NE = GetScl(max_NE);
    scl_NVl = GetScl(max_NVl);
    scl_NElg = GetScl(max_NElg);
    NV_gh = CheckGhost(); // + NVl;;
    NV = NV_gh + NVl;
    elist = (edge*)malloc(sizeof(edge) * (NE));
    M_v = (long*)malloc(sizeof(long) * (NV));
    assert(M_v);
    assert(elist);
    memset(M_v, 0, sizeof(long) * (NV));
    NE_list_all = 0;
    NE_list_ll = 0;
    NE_list_gl = 0;
    NE_list_gg = 0;
    NV_list_all = 0;
    NV_list_l = 0;
    NV_list_g = 0;
    for (int p = 0; p < num_par; p++) {
        GLV* G_src = par_src[p];
        for (int v = 0; v < G_src->NVl; v++) {
            long base_src = off_src[p];
            long base_lved = off_lved[p];
            long c_src = G_src->C[v];
            long c_mg;
            if (c_src < par_lved[p]->NVl)
                c_mg = c_src + base_lved;
            else
                c_mg = p_v_new[p][c_src];
            G_src->C[v] = c_mg;
// M_v[v+base_lved] = c_mg;
#ifdef DBG_PAR_PRINT
            printf("DBGPREMG:p=%d  v=%d base_src=%d  base_lved=%d C1=%d, isLocal%d, c_mg=%d\n", p, v, base_src,
                   base_lved, G_src->C[v], c_src < par_lved[p]->NVl, c_mg);
#endif
        }
    }
    st_PreMerged = true;
}
int ParLV::AddGLV(GLV* plv) {
    assert(plv);
    par_src[num_par] = plv;
    num_par++;
    return num_par;
}
long ParLV::FindGhostInLocalC(long me) {
    long e_org = -me - 1;
    int idx = 0;
    // 1. find #p
    for (int p = 0; p < num_par; p++) {
        if (off_src[p] <= e_org && e_org < off_src[p + 1]) {
            idx = p;
            break;
        }
    }
    // 2.
    long address = e_org - off_src[idx];
    long m_src = par_src[idx]->M[address];
    // assert(m_org ==  m_src);
    long c_src = par_src[idx]->C[address];
    long c_src_m = c_src + off_lved[idx];
    printf("e_org=%-4ld - %-4ld = address:%-4ld; c_src:%-4ld+off%-4ld=c_src_m%-4ld\n", e_org, off_src[idx], address,
           c_src, off_lved[idx], c_src_m);
    return c_src_m;
}
int ParLV::FindParIdx(long e_org) {
    int idx = 0;
    // 1. find #p
    for (int p = 0; p < num_par; p++) {
        if (off_src[p] <= e_org && e_org < off_src[p + 1]) {
            idx = p;
            break;
        }
    }
    return idx;
}
int ParLV::FindParIdxByID(int id) {
    if (!this->st_Partitioned) return -1;
    for (int p = 0; p < num_par; p++)
        if (this->par_lved[p]->ID == id) return p;
    if (!this->st_ParLved) return -1;
    for (int p = 0; p < num_par; p++)
        if (this->par_src[p]->ID == id) return p;
    return -1;
}
pair<long, long> ParLV::FindCM_1hop(int idx, long e_org) {
    // 2.
    pair<long, long> ret;
    long addr_v = e_org - off_src[idx];
    long c_src_sync = par_src[idx]->C[addr_v];
    long c_lved_new = c_src_sync; // key logic
    long m_lved_new = par_lved[idx]->M[c_lved_new];
    ret.first = c_lved_new;
    ret.second = m_lved_new;
    return ret;
}

pair<long, long> ParLV::FindCM_1hop(long e_org) {
    // 2.
    int idx = FindParIdx(e_org);
    pair<long, long> ret;
    long addr_v = e_org - off_src[idx];
    long c_src_sync = par_src[idx]->C[addr_v];
    long c_lved_new = c_src_sync; // key logic
    long m_lved_new = par_lved[idx]->M[c_lved_new];
    ret.first = c_lved_new;
    ret.second = m_lved_new;
    return ret;
}
long ParLV::FindC_nhop(long m_g) {
    assert(m_g < 0);
    long m_next = m_g;
    int cnt = 0;

    do {
        long e_org = -m_next - 1;
        int idx = FindParIdx(e_org);
        long v_src = e_org - off_src[idx]; // dbg
        pair<long, long> cm = FindCM_1hop(idx, e_org);
        long c_lved_new = cm.first;
        long m_lved_new = cm.second;
        /*
                        //debug begin
                        printf("DBG:FindC:cnt=%d, m:%-4d --> e_org:%-4d, idx:%-2d, --> v_src:%-4d, c_src&lved:%-4d,
           m_lved:%-4d --> c_new:%d",
                                                  cnt,  m_next,   e_org,      idx,          v_src,     c_lved_new,
           m_lved_new, c_lved_new + off_lved[idx]);
                        if(m_lved_new>=0)
                                printf("-> c_new:%d\n",c_lved_new + off_lved[idx]);
                        else
                                printf("\n");
                        //debug end
                         *
                         */
        cnt++;

        if (m_lved_new >= 0)
            return c_lved_new + off_lved[idx];
        else if (m_lved_new == m_g) {
            return m_g;
        } else { // m_lved_new<0;
            m_next = m_lved_new;
        }

    } while (cnt < 2 * num_par);
    return m_g; // no local community for the ghost which should be add as a new community
}

//#define DBG_PAR_PRINT
long FindOldOrAddNew(map<long, long>& map_v, long& NV, long v) {
    map<long, long>::iterator iter;
    int ret;
    iter = map_v.find(v);
    if (iter == map_v.end()) {
        ret = NV++; // add new
#ifdef DBG_PAR_PRINT
        printf("DBG_PAR_PRINT, new:%d ", ret);
#endif
    } else {
        ret = iter->second; // find old
#ifdef DBG_PAR_PRINT
        printf("DBG_PAR_PRINT, old:%d ", ret);
#endif
    }
    return ret;
}
long ParLV::CheckGhost() {
    long NV_gh_new = 0;
    for (int p = 0; p < num_par; p++) {
        GLV* G_src = par_src[p];
        GLV* G_lved = par_lved[p];
        long* vtxPtr = G_lved->G->edgeListPtrs;
        edge* vtxInd = G_lved->G->edgeList;
        p_v_new[p] = (long*)malloc(sizeof(long) * (G_lved->NV));
        assert(p_v_new[p]);
        for (int v = G_lved->NVl; v < G_lved->NV; v++) {
            long mv = G_lved->M[v];
            long v_new = FindC_nhop(mv);
            if (v_new == mv) {
                p_v_new[p][v] = FindOldOrAddNew(m_v_gh, NV_gh_new, v_new) + this->NVl;
#ifdef DBG_PAR_PRINT
                printf("CheckGhost: p=%-2d  v=%-6d mv=%-6d  v_new=%-6d NV_gh=%d\n", p, v, mv, p_v_new[p][v], NV_gh_new);
#endif
            } else {
                p_v_new[p][v] = v_new;
#ifdef DBG_PAR_PRINT
                printf("CheckGhost: p=%-2d  v=%-6d mv=%-6d  v_new=%-6d  isNVL%d\n", p, v, mv, v_new, v_new < this->NVl);
#endif
            }
        }
    }
    return NV_gh_new;
}
long ParLV::MergingPar2_ll() {
    // 1.create new edge list;
    long num_e_dir = 0;
    NEll = 0;
    NEself = 0;
    // long num_c_g   = 0;
    for (int p = 0; p < num_par; p++) {
        GLV* G_src = par_src[p];
        /*for(int v=0; v<G_src->NVl; v++){
                long off_local = off_lved[p];
                long c_src = G_src->C[v];
                if(c_src < par_lved[p]->NVl)
                        c_src += off_local;
                else
                        c_src = p_v_new[p][c_src];
                G_src->C[v] = c_src;
        }*/
        GLV* G_lved = par_lved[p];
        long* vtxPtr = G_lved->G->edgeListPtrs;
        edge* vtxInd = G_lved->G->edgeList;
        for (int v = 0; v < G_lved->NVl; v++) {
            assert(G_lved->M[v] >= 0);
            long adj1 = vtxPtr[v];
            long adj2 = vtxPtr[v + 1];
            int degree = adj2 - adj1;
            long off_local = off_lved[p];
            long v_new = v + off_local;
            long e_new;
            p_v_new[p][v] = v_new;
            // M_v[v_new] = v_new;
            for (int d = 0; d < degree; d++) {
                long e = vtxInd[adj1 + d].tail;
                long me = G_lved->M[e];
                bool isGhost = me < 0;
                if (v < e || isGhost) continue;
                e_new = e + off_local;
                double w = vtxInd[adj1 + d].weight;
                elist[num_e_dir].head = v_new;
                elist[num_e_dir].tail = e_new;
                elist[num_e_dir].weight = w;
                assert(v_new < this->NVl);
                assert(e_new < this->NVl);
                num_e_dir++;
                NEll++;
                if (v_new == e_new) NEself++;
#ifdef DBG_PAR_PRINT
                printf(
                    "LOCAL: p=%-2d v=%-8ld mv=%-8ld v_new=%-8ld, e=%-8ld me=%-8ld e_new=%-8ld w=%-3.0f NE=%-8ld "
                    "NEself=%-8ld NEll=%-8ld \n",
                    p, v, 0, v_new, e, me, e_new, w, num_e_dir, NEself, NEll);
#endif
                // AddEdge(elist, v, e, w, M_v);
            } // for d
        }     // for v
    }
    st_Merged_ll = true;
    return num_e_dir;
}
long ParLV::MergingPar2_gh() {
    long num_e_dir = 0;

    for (int p = 0; p < num_par; p++) {
        GLV* G_src = par_src[p];
        GLV* G_lved = par_lved[p];
        long* vtxPtr = G_lved->G->edgeListPtrs;
        edge* vtxInd = G_lved->G->edgeList;

        for (int v = G_lved->NVl; v < G_lved->NV; v++) {
            long mv = G_lved->M[v]; // should be uniqe in the all sub-graphNew
            assert(mv < 0);
            // combination of possible connection:
            // 1.1   C(head_gh)==Normal C and tail is local;
            // <local,
            // local>
            // 1.2   C(head_gh)==Normal C and tail is head_gh itself;
            // <local,
            // local>
            // 1.3.1 C(head_gh)==Normal C,and tail is other ghost in current sub graphNew and its C is normal
            // <local,
            // local>
            // 1.3.2 C(head_gh)==Normal C,and tail is other ghost in current sub graphNew and its C is other ghost
            // <local, m(tail_ghost)>,
            // 2     C(head_gh) is other ghost

            long v_new = p_v_new[p][v]; /*
         if(v_new <0 ){
                 if(isOnlyGL)
                         continue;
                 v_new = FindOldOrAddNew(m_v_gh, this->NV_gh, v_new);
                 v_new += this->NVl;// Allocated a new community for local
         }*/

            long adj1 = vtxPtr[v];
            long adj2 = vtxPtr[v + 1];
            int degree = adj2 - adj1;
            long off_local = off_lved[p];
            // trace v

            for (int d = 0; d < degree; d++) {
                double w = vtxInd[adj1 + d].weight;
                long e = vtxInd[adj1 + d].tail;
                long me = G_lved->M[e];
                bool isGhost = me < 0;
                if (v < e) continue;
                long e_new;
                if (me >= 0)
                    e_new = e + off_local;
                else if (me == mv) {
                    // assert (me == mv);
                    e_new = v_new;
                    NEself++;
                    NEgg++;
                } else {
                    e_new = p_v_new[p][e];
                }
                elist[NEll + num_e_dir].head = v_new;
                elist[NEll + num_e_dir].tail = e_new;
                elist[NEll + num_e_dir].weight = w;
                // M_v[v_new] = v_new;
                num_e_dir++;
#ifdef DBG_PAR_PRINT
                printf(
                    "GHOST: p=%-2d v=%-8ld mv=%-8ld v_new=%-8ld, e=%-8ld me=%-8ld e_new=%-8ld w=%-3.0f NE=%-8ld "
                    "NEself=%-8ld NEgg=%-8ld \n",
                    p, v, mv, v_new, e, me, e_new, w, num_e_dir, NEself, NEgg);
#endif
            }
            // 1 findDesC;
        }
    } // for sub graphNew ;

    st_Merged_gh = true;
    return num_e_dir;
}
GLV* ParLV::MergingPar2(int& id_glv) {
    long num_e_dir = 0;
    // CheckGhost();
    num_e_dir += MergingPar2_ll();
    num_e_dir += MergingPar2_gh();
    NE = num_e_dir;

    graphNew* Gnew = (graphNew*)malloc(sizeof(graphNew));
    GLV* glv = new GLV(id_glv);
    glv->SetName_ParLvMrg(num_par, plv_src->ID);
    printf("\033[1;37;40mINFO: PAR\033[0m: NV( %-8d) = NVl ( %-8d) + NV_gh( %-8d) \n", NV, NVl, NV_gh);
    printf(
        "\033[1;37;40mINFO: PAR\033[0m: NE( %-8d) = NEll( %-8d) + NElg ( %-8d) + NEgl( %-8d) + NEgg( %-8d) + NEself( "
        "%-8d) \n",
        NE, NEll, NElg, NEgl, NEgg, NEself);
    GetGFromEdge_selfloop(Gnew, elist, this->NVl + this->NV_gh, num_e_dir);
    glv->SetByOhterG(Gnew);
    // glv->SetM(M_v);
    st_Merged = true;
    plv_merged = glv;
    return glv;
}

GLV* ParLV::FinalLouvain(char* opts_xclbinPath,
                         int numThreads,
                         int& id_glv,
                         long minGraphSize,
                         double threshold,
                         double C_threshold,
                         bool isParallel,
                         int numPhase) {
    if (st_Merged == false) return NULL;
    bool hasGhost = false;
    plv_final = LouvainGLV_general(hasGhost, this->flowMode, 0, plv_merged, opts_xclbinPath, numThreads, id_glv,
                                   minGraphSize, threshold, C_threshold, isParallel, numPhase);
    printf("\033[1;37;40mINFO: PAR\033[0m: Community of plv_merged is updated\n");
    // assert(plv_merged->NV==plv_src->NV);
    for (int p = 0; p < num_par; p++) {
        for (long v_sub = 0; v_sub < par_src[p]->NVl; v_sub++) {
            long v_orig = v_sub + off_src[p];
            long v_merged = par_src[p]->C[v_sub];
            plv_src->C[v_orig] = plv_merged->C[v_merged];
#ifdef DBG_PAR_PRINT
            printf("DBG_FINAL: p=%d  v_sub=%d v_orig=%d v_final=%d C_final=%d\n", p, v_sub, v_orig, v_merged,
                   plv_src->C[v_orig]);
#endif
        }
    }
    printf("\033[1;37;40mINFO: PAR\033[0m: Community of plv_src is updated\n");
    st_FinalLved = true;
    return plv_final;
}
double ParLV::TimeStar() {
    return timesPar.time_star = omp_get_wtime();
}
double ParLV::TimeDonePar() {
    return timesPar.time_done_par = omp_get_wtime();
}
double ParLV::TimeDoneLv() {
    return timesPar.time_done_lv = omp_get_wtime();
}
double ParLV::TimeDonePre() {
    return timesPar.time_done_pre = omp_get_wtime();
}
double ParLV::TimeDoneMerge() {
    return timesPar.time_done_mg = omp_get_wtime();
}
double ParLV::TimeDoneFinal() {
    timesPar.time_done_fnl = omp_get_wtime();
    return timesPar.time_done_fnl;
}
double ParLV::TimeAll_Done() {
    timesPar.timePar_all = timesPar.time_done_par - timesPar.time_star;
    timesPar.timeLv_all = timesPar.time_done_lv - timesPar.time_done_par;
    timesPar.timePre = timesPar.time_done_pre - timesPar.time_done_lv;
    timesPar.timeMerge = timesPar.time_done_mg - timesPar.time_done_pre;
    timesPar.timeFinal = timesPar.time_done_fnl - timesPar.time_done_mg;
    timesPar.timeAll = timesPar.time_done_fnl - timesPar.time_star;
    return timesPar.timeAll;
}
void ParLV::PrintTime() {
    printf("\033[1;37;40mINFO\033[0m: Total time for partition orignal       : %lf\n", timesPar.timePar_all);
    printf("\033[1;37;40mINFO\033[0m: Total time for partition Louvain subs  : %lf\n", timesPar.timeLv_all);
    for (int d = 0; d < num_dev; d++) { // for parlv.timeLv_dev[d]
        printf("\033[1;37;40m    \033[0m: Total time for Louvain on dev-%1d        : %lf\t = ", d,
               timesPar.timeLv_dev[d]);
        for (int p = d; p < num_par; p += num_dev) printf("+ %3.4f ", timesPar.timeLv[p]);
        printf("\n");
    }
    printf("\033[1;37;40mINFO\033[0m: Total time for partition pre-Merge     : %lf\n", timesPar.timePre);
    printf("\033[1;37;40mINFO\033[0m: Total time for partition Merge         : %lf\n", timesPar.timeMerge);
    printf("\033[1;37;40mINFO\033[0m: Total time for partition Final Louvain : %lf\n", timesPar.timeFinal);
    printf("\033[1;37;40mINFO\033[0m: Total time for partition All flow      : %lf\n", timesPar.timeAll);
}
void ParLV::PrintTime2() {
    // Final number of clusters       : 225
    // Final modularity               :
    printf("\033[1;37;40mINFO\033[0m: Final number of clusters               : %ld\n", plv_src->com_list.back().NC);
    printf("\033[1;37;40mINFO\033[0m: Final modularity                       : %lf\n", plv_src->com_list.back().Q);
    printf("\033[1;37;40mINFO\033[0m: Total time for partition + Louvain     : %lf\n", timesPar.timePar_all);
    printf("\033[1;37;40mINFO\033[0m: Total time for partition pre-Merge     : %lf\n", timesPar.timePre);
    printf("\033[1;37;40mINFO\033[0m: Total time for partition Merge         : %lf\n", timesPar.timeMerge);
    printf("\033[1;37;40mINFO\033[0m: Total time for partition Final Louvain : %lf\n", timesPar.timeFinal);
    printf("\033[1;37;40mINFO\033[0m: Total time for partition All flow      : %lf\n", timesPar.timeAll);
}
void ParLV::CleanTmpGlv() {
    for (int p = 0; p < num_par; p++) {
        delete (par_src[p]);
        delete (par_lved[p]);
    }
    delete (plv_merged);
}
///////////////////////////////////////////////////////////////////////////
GLV* par_general(GLV* src, SttGPar* pstt, int& id_glv, long start, long end, bool isPrun, int th_prun) {
    GLV* des;
    if (isPrun) {
        printf("\033[1;37;40mINFO\033[0m: Partition of \033[1;31;40mPruning\033[0m is used and th_maxGhost=%d \n",
               th_prun);
        des = pstt->ParNewGlv_Prun(src->G, start, end, id_glv, th_prun);
    } else {
        printf("\033[1;37;40mINFO\033[0m: Partition of \033[1;31;40mNoraml\033[0m is used\n");
        des = pstt->ParNewGlv(src->G, start, end, id_glv);
    }
    des->SetName_par(des->ID, src->ID, start, end, isPrun ? 0 : th_prun);
    pstt->PrintStt();
    return des;
}

GLV* par_general(GLV* src, int& id_glv, long start, long end, bool isPrun, int th_prun) {
    SttGPar stt;
    return par_general(src, &stt, id_glv, start, end, isPrun, th_prun);
}

GLV* LouvainGLV_general_par(int flowMode,
                            ParLV& parlv,
                            char* xclbinPath,
                            int numThreads,
                            int& id_glv,
                            long minGraphSize,
                            double threshold,
                            double C_threshold,
                            bool isParallel,
                            int numPhase) {
    parlv.TimeStar();
    long vsize = parlv.plv_src->NV / parlv.num_par;
    long start = 0;
    long end = start + vsize;
    // ParLV parlv;
    // parlv.Init(glv_src, num_par, num_dev);//should never release resource and object who pointed; Just work as a
    // handle
    for (int p = 0; p < parlv.num_par; p++) {
        double time_par = omp_get_wtime();
        GLV* tmp = par_general(parlv.plv_src, &(parlv.stt[p]), id_glv, start, end, parlv.isPrun, parlv.th_prun);
        parlv.timesPar.timePar[p] = omp_get_wtime() - time_par;
        parlv.par_src[p] = tmp;
        start = end;
        end = (p == parlv.num_par - 2) ? parlv.plv_src->NV : start + vsize;
    }
    parlv.st_Partitioned = true;
    parlv.TimeDonePar();

#ifndef _SINGLE_THREAD_MULTI_DEV_
    std::thread td[parlv.num_dev];
    {
        for (int dev = 0; dev < parlv.num_dev; dev++) {
            parlv.timesPar.timeLv_dev[dev] = omp_get_wtime();
            bool hasGhost = true;
            td[dev] = std::thread(LouvainGLV_general_batch_thread, hasGhost, flowMode, dev, id_glv, parlv.num_dev,
                                  parlv.num_par, parlv.timesPar.timeLv, parlv.par_src, parlv.par_lved, xclbinPath,
                                  numThreads, minGraphSize, threshold, C_threshold, isParallel, numPhase);
        }

        for (int dev = 0; dev < parlv.num_dev; dev++) {
            td[dev].join();
            parlv.timesPar.timeLv_dev[dev] = omp_get_wtime() - parlv.timesPar.timeLv_dev[dev];
        }
    }
#else
    {
        for (int dev = 0; dev < parlv.num_dev; dev++)
            parlv.timeLv_dev[dev] = LouvainGLV_general_batch(
                dev, parlv.num_dev, parlv.num_par, parlv.timeLv, parlv.par_src, parlv.par_lved, xclbinPath, numThreads,
                id_glv, minGraphSize, threshold, C_threshold, isParallel, numPhase);
    }
#endif
    parlv.st_ParLved = true;

    parlv.TimeDoneLv();

    parlv.PreMerge();

    parlv.TimeDonePre();

    parlv.MergingPar2(id_glv);

    parlv.TimeDoneMerge();

    GLV* glv_final =
        parlv.FinalLouvain(xclbinPath, numThreads, id_glv, minGraphSize, threshold, C_threshold, isParallel, numPhase);

    parlv.TimeDoneFinal();

    return glv_final;
}

GLV* LouvainGLV_general_par_OneDev(int flowMode,
                                   ParLV& parlv,
                                   char* xclbinPath,
                                   int numThreads,
                                   int& id_glv,
                                   long minGraphSize,
                                   double threshold,
                                   double C_threshold,
                                   bool isParallel,
                                   int numPhase) {
    parlv.TimeStar();
    long vsize = parlv.plv_src->NV / parlv.num_par;
    long start = 0;
    long end = start + vsize;
    // ParLV parlv;
    // parlv.Init(glv_src, num_par, num_dev);//should never release resource and object who pointed; Just work as a
    // handle
    for (int p = 0; p < parlv.num_par; p++) {
        double time_par = omp_get_wtime();
        GLV* tmp = par_general(parlv.plv_src, &(parlv.stt[p]), id_glv, start, end, parlv.isPrun, parlv.th_prun);
        parlv.timesPar.timePar[p] = omp_get_wtime() - time_par;
        parlv.par_src[p] = tmp;
        start = end;
        end = (p == parlv.num_par - 2) ? parlv.plv_src->NV : start + vsize;
    }
    parlv.st_Partitioned = true;
    parlv.TimeDonePar();

    const int id_dev = 0;
    printf("INFO: using one device(%2d) for Louvain \n", id_dev);
    parlv.timesPar.timeLv_dev[id_dev] = omp_get_wtime();
    for (int p = 0; p < parlv.num_par; p++) {
        double time1 = omp_get_wtime();
        int id_glv_dev = id_glv + p;
        bool hasGhost = true;
        GLV* glv_t = LouvainGLV_general(hasGhost, flowMode, id_dev, parlv.par_src[p], xclbinPath, numThreads,
                                        id_glv_dev, minGraphSize, threshold, C_threshold, isParallel, numPhase);
        parlv.par_lved[p] = glv_t;
        // pushList(glv_t);
        parlv.timesPar.timeLv[p] = omp_get_wtime() - time1;
    }
    parlv.timesPar.timeLv_dev[id_dev] = omp_get_wtime() - parlv.timesPar.timeLv_dev[id_dev];

    parlv.st_ParLved = true;

    parlv.TimeDoneLv();

    parlv.PreMerge();

    parlv.TimeDonePre();

    parlv.MergingPar2(id_glv);

    parlv.TimeDoneMerge();

    GLV* glv_final =
        parlv.FinalLouvain(xclbinPath, numThreads, id_glv, minGraphSize, threshold, C_threshold, isParallel, numPhase);

    parlv.TimeDoneFinal();

    return glv_final;
}

GLV* LouvainGLV_general_par_OneDev_forl3(int flowMode,
                                         ParLV& parlv,
                                         char* xclbinPath,
                                         int numThreads,
                                         int& id_glv,
                                         long minGraphSize,
                                         double threshold,
                                         double C_threshold,
                                         bool isParallel,
                                         int numPhase) {
    parlv.TimeStar();
    long vsize = parlv.plv_src->NV / parlv.num_par;

    // ParLV parlv;
    // parlv.Init(glv_src, num_par, num_dev);//should never release resource and object who pointed; Just work as a
    // handle

    parlv.st_Partitioned = true;
    parlv.TimeDonePar();
    const int id_dev = 0;

    parlv.timesPar.timeLv_dev[id_dev] = omp_get_wtime();

#pragma omp parallel for
    for (int p = 0; p < parlv.num_par; p++) {
        long start = vsize * p;
        long end = (p == parlv.num_par - 1) ? parlv.plv_src->NV : (start + vsize);

        printf("INFO: start 1 partition for Louvain \n");
        double time_par = omp_get_wtime();
        GLV* tmp = par_general(parlv.plv_src, &(parlv.stt[p]), id_glv, start, end, parlv.isPrun, parlv.th_prun);
        parlv.timesPar.timePar[p] = omp_get_wtime() - time_par;
        parlv.par_src[p] = tmp;
        // start = end;
        // end = (p==parlv.num_par-2)?parlv.plv_src->NV:start + vsize;
        printf("INFO: using one device(%2d) for Louvain \n", id_dev);

        double time1 = omp_get_wtime();
        int id_glv_dev = id_glv + p;
        bool hasGhost = true;
        GLV* glv_t = LouvainGLV_general(hasGhost, flowMode, id_dev, parlv.par_src[p], xclbinPath, numThreads,
                                        id_glv_dev, minGraphSize, threshold, C_threshold, isParallel, numPhase);
        parlv.par_lved[p] = glv_t;
        // pushList(glv_t);
        parlv.timesPar.timeLv[p] = omp_get_wtime() - time1;
    }

    //	for(int p=0; p<parlv.num_par; p++){
    //		double time1 = omp_get_wtime();
    //		int id_glv_dev= id_glv+p;
    //		bool hasGhost=true;
    //		GLV* glv_t = LouvainGLV_general(hasGhost, flowMode, id_dev, parlv.par_src[p],  xclbinPath,  numThreads,
    // id_glv_dev, minGraphSize, threshold, C_threshold, isParallel, numPhase);
    //		parlv.par_lved[p] = glv_t;
    //		//pushList(glv_t);
    //		parlv.timesPar.timeLv[p] = omp_get_wtime() - time1;
    //	}
    parlv.timesPar.timeLv_dev[id_dev] = omp_get_wtime() - parlv.timesPar.timeLv_dev[id_dev];

    parlv.st_ParLved = true;

    parlv.TimeDoneLv();

    parlv.PreMerge();

    parlv.TimeDonePre();

    parlv.MergingPar2(id_glv);

    parlv.TimeDoneMerge();

    GLV* glv_final =
        parlv.FinalLouvain(xclbinPath, numThreads, id_glv, minGraphSize, threshold, C_threshold, isParallel, numPhase);

    parlv.TimeDoneFinal();

    return glv_final;
}

GLV* LouvainGLV_general_par(int mode,
                            GLV* glv_orig,
                            int num_par,
                            int num_dev,
                            int isPrun,
                            int th_prun,
                            char* xclbinPath,
                            int numThreads,
                            int& id_glv,
                            long minGraphSize,
                            double threshold,
                            double C_threshold,
                            bool isParallel,
                            int numPhase) {
    ParLV parlv;
    const int n_dev = 1;
    parlv.Init(mode, glv_orig, num_par, n_dev, isPrun, th_prun);
    GLV* glv_final = LouvainGLV_general_par_OneDev_forl3(mode, parlv, xclbinPath, numThreads, id_glv, minGraphSize,
                                                         threshold, C_threshold, isParallel, numPhase);
    parlv.PrintSelf();
    parlv.TimeAll_Done();
    parlv.PrintTime();
    parlv.CleanTmpGlv();
    // To update original graphNew with new number of community NC and modularity Q
    parlv.plv_src->NC = glv_final->NC;
    parlv.plv_src->PushFeature(0, 0, 0.0, true);
    return glv_final;
}

void ParLV_general_batch_thread(int flowMode,
                                GLV* plv_orig,
                                int id_dev,
                                int num_dev,
                                int num_par,
                                double* timeLv,
                                GLV* par_src[],
                                GLV* par_lved[],
                                char* xclbinPath,
                                int numThreads,
                                long minGraphSize,
                                double threshold,
                                double C_threshold,
                                bool isParallel,
                                int numPhase) {
    long vsize = plv_orig->NV / num_par;
    GLV* glv_t;
    int id_glv = id_dev * 64;

    for (int p = id_dev; p < num_par; p += num_dev) {
        double time_par = omp_get_wtime();
        long start = p * vsize;
        long end = (p == num_par - 1) ? plv_orig->NV : start + vsize;
        GLV* tmp = par_general(plv_orig, id_glv, start, end, true, 1);
        par_src[p] = tmp;
        start = end;
    }
    for (int p = id_dev; p < num_par; p += num_dev) {
        double time1 = omp_get_wtime();
        glv_t = LouvainGLV_general(true, flowMode, 0, par_src[p], xclbinPath, numThreads, id_glv, minGraphSize,
                                   threshold, C_threshold, isParallel, numPhase);
        par_lved[p] = glv_t;
        // pushList(glv_t);
        timeLv[p] = omp_get_wtime() - time1;
    }
}
