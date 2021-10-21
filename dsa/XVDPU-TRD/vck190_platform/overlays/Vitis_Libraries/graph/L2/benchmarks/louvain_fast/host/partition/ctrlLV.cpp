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
#include "ctrlLV.h"
#include "partitionLouvain.hpp"
#include "louvainPhase.h"
#include "ParLV.h"
#include <thread>

TimeStt::TimeStt(char* nm) {
    name = nm;
    time = omp_get_wtime();
}
double TimeStt::End() {
    time = omp_get_wtime() - time;
    return time;
};
void TimeStt::Print() {
    printf("TIME_STT: %s =%lf sec\n", name, time);
};
void TimeStt::EndPrint() {
    End();
    Print();
};

myCmd::myCmd() {
    cnt_his = 0;
    argc = 0;
    cmd_last = -1;
};
char* myCmd::cmd_SkipSpace(char* str) {
    char* pch = str;
    while (*pch != 0) {
        if (*pch == ' ' || *pch == '\t') {
            pch++;
        } else
            break;
    }
    return pch;
}
bool isChNormal(char ch) {
    return (ch != '\0') && (ch != '\n') && (ch != ' ') && (ch != '\t' && (ch != '#'));
}
int myCmd::cmd_CpyWord(char* des, char* src) {
    int cnt = 0;
    assert(des);
    assert(src);
    while (isChNormal(*src)) {
        *des = *src;
        src++;
        des++;
        cnt++;
    }
    *des = '\0';
    return cnt;
}
void myCmd::cmd_resetArg() {
    if (argc == 0) return;
    for (int i = 0; i < argc; i++) argv[i][0] = '\0';
    argc = 0;
}

int myCmd::cmd_findPara(char* para) {
    for (int i = 1; i < argc; i++) {
        if (0 == strcmp(argv[i], para)) return i;
    }
    return -1;
}

int Getline(char* str) {
    char ch;
    int cnt = 0;
    do {
        ch = getchar();
        str[cnt++] = ch;
    } while (ch != '\n');
    return cnt;
}

int myCmd::cmd_Getline() { //( char** argv){
    // int argc;
    // assert(argv!=0);
    char str[4096];

    cmd_resetArg();
    // printf("$LVCMD:\/");
    printf("\033[1;31;40m$\033[0m\033[1;34;40m[LVCMD]$:\ \033[0m");
    // scanf("%[^\n]", str);
    Getline(str);
    line_last = str;
    // his_cmd.push_back(line_last);
    char* pch = str;
    do {
        pch = cmd_SkipSpace(pch);
        if (isChNormal(*pch)) {
            pch += cmd_CpyWord(argv[argc++], pch);
        }
    } while (*pch != '\n' && *pch != '\0' && *pch != '#');
    return argc;
}
int myCmd::cmd_Getline(char* str) { //( char** argv){
    cmd_resetArg();
    line_last = str;
    // his_cmd.push_back(line_last);
    char* pch = str;
    do {
        pch = cmd_SkipSpace(pch);
        if (isChNormal(*pch)) {
            pch += cmd_CpyWord(argv[argc++], pch);
        }
    } while (*pch != '\n' && *pch != '\0' && *pch != '#');
    return argc;
}
int myCmd::cmd_GetCmd() {
    cmd_Getline();
    if (argc == 0) {
        printf("\033[1;31;40mERROR\033[0m: No valid command\n");
        return -1;
    }
    for (cmd_last = 0; cmd_last < NUM_CMD; cmd_last++) {
        if (0 == strcmp(cmdlist[cmd_last], argv[0])) break;
    }
    if (cmd_last < 0 && cmd_last == NUM_CMD) {
        printf("\033[1;31;40mERROR\033[0m: No matched command for %s \n", argv[0]);
        cmd_last = -1;
    }
    return cmd_last;
}
int myCmd::cmd_GetCmd(char* str) {
    cmd_Getline(str);
    if (argc == 0) {
        printf("\033[1;31;40mERROR\033[0m: No valid command\n");
        return -1;
    }
    for (cmd_last = 0; cmd_last < NUM_CMD; cmd_last++) {
        if (0 == strcmp(cmdlist[cmd_last], argv[0])) break;
    }
    if (cmd_last < 0 && cmd_last == NUM_CMD) {
        printf("\033[1;31;40mERROR\033[0m: No matched command for %s \n", argv[0]);
        cmd_last = -1;
    }
    return cmd_last;
}
int myCmd::PrintHistory() {
    list<string>::iterator iter;
    iter = his_cmd.begin();
    int idx = cmd_findPara("-f");
    char* fileName = argv[idx + 1];
    int cnt = 0;
    if (idx < 0) {
        while (iter != his_cmd.end()) {
            printf("%8d: %s", cnt++, (*iter).data());
            iter++;
        }
        printf("\n");
        return 0;
    }
    if (argc < 2) {
        printf("\033[1;31;40mERROR\033[0m: Lack <file> for -f \n");
        return -1;
    }
    FILE* file = fopen(fileName, "w");
    if (file == NULL) {
        printf("\033[1;31;40mERROR\033[0m: Cannot open the batch file: %s\n", fileName);
        return -1;
    }
    while (iter != his_cmd.end()) {
        printf("%8d: %s", cnt++, (*iter).data());
        fprintf(file, "%s", (*iter).data());
        iter++;
    }
    printf("\n");
    fclose(file);
    return 0;
}
void CtrlLouvain::PrintMan() {
    printf(
        "\033[1;37;40mWelcome to use this command line for running Louvain, the following commands can be "
        "used:\033[0m:: \n");
    for (int i = 0; i < NUM_CMD; i++) {
        printf("CMD %2d\t %s\n", i, cmdlist_intro[i]);
    }
}
CtrlLouvain::CtrlLouvain() {
    // printf("\033[1;37;40mINFO\033[0m:: \n");
    minGraphSize = 10;
    threshold = 0.000001; // default 0.0001
    C_threshold = 0.0001; // default 0.000001
    coloring = 1;         // default =1;
    numThreads = 1;       // default =1;
    glv_curr = NULL;
    glv_temp = NULL;
    cnt_list = 0;
    id_glv = 0;
    isParallel = false;
    flowMode = MD_NORMAL;
    useKernel = false;
    numPhase = 100;
    num_dev = 0;
    glv_list.clear();
}
bool CtrlLouvain::IsTempInLIst() {
    if (glv_temp != NULL) {
        return IsInList(glv_temp);
    } else
        return false;
}
void CtrlLouvain::ShowPara() {
    printf("\n");
    printf(
        "=====[ Parameters for Louvain "
        "]============================================================================================================="
        "=================\n");
    printf("| \033[1;32;40m minGraphSize\033[0m: %-4d\n", minGraphSize); //
    printf("| \033[1;32;40m threshold   \033[0m: %f\n", threshold);      //= 0.000001;   //default 0.000001
    printf("| \033[1;32;40m C_thresholt \033[0m: %f\n", C_threshold);    //= 0.0001; //default 0.0001
    // printf("| \033[1;32;40m isParallel  \033[0m: %s\n", isParallel?"true":"false"); //= 0.0001; //default 0.0001
    printf("| \033[1;32;40m numThread \033[0m  : %d\n", numThreads); //= 0.0001; //default 0.0001 xclbinPath
    printf("| \033[1;32;40m xclbinPath \033[0m : %s\t\t\t", xclbinPath);
    if (this->flowMode == MD_CLASSIC) printf("| \033[1;32;40m Golden Flow \033[0m \n");
    if (this->flowMode == MD_NORMAL) printf("| \033[1;32;40m Support partition \033[0m \n");
    if (this->flowMode == MD_FAST) printf("| \033[1;32;40m Support pruning for fast iteration \033[0m \n");
    printf("| \033[1;32;40m numDevice \033[0m  : %d\n", num_dev);
    // printf("coloring\t%d\n", coloring);    //= 1;    //default =1;
    // printf("numThreads\t%8x\t", numThreads);  //= 1;	//default =1;
    printf(
        "|----[ Active Object "
        "]-------------------------------------------------------------------------------------------------------------"
        "--------------------------\n");
    printf("| \033[1;36;40m[glv_curr]\033[0m\t");
    if (glv_curr)
        glv_curr->printSimple();
    else
        printf("NULL\n");
    if (IsTempInLIst()) {
        printf("| \033[1;36;40m[glv_temp]\033[0m\t");
        if (glv_temp)
            glv_temp->printSimple();
        else
            printf("NULL\n");
    } else {
        printf("| \033[1;36;40m[glv_temp]*\033[0m\t");
        if (glv_temp)
            glv_temp->printSimple();
        else
            printf("NULL\n");
    }
    printf(
        "|----[ List Status "
        "]-------------------------------------------------------------------------------------------------------------"
        "----------------------------\n");
    printf("| \033[1;35;40m[list_glv]\033[0m\t%d glv(s) in list\n", cnt_list); //= 0;
    printf("| \033[1;35;40m[ ParLV  ]\033[0m\t%s \t %s \t %s \t %s\n", parlv.st_Partitioned ? "Partitioned" : "Empty",
           parlv.st_ParLved ? "ParLved" : "Not ParLved", parlv.st_PreMerged ? "PreMerged" : "Not PreMerged",
           parlv.st_Merged ? "Merged" : "Not Merged", parlv.st_FinalLved ? "st_FinalLved" : "Not st_FinalLved");
    printf(
        "=============================================================================================================="
        "===============================================\n");
}
void CtrlLouvain::CleanList() {
    list<GLV*>::iterator iter;
    iter = glv_list.begin();
    while (iter != glv_list.end()) {
        printf("\033[1;37;40mINFO\033[0m: Deleting CtrlLouvain ID:%d name:%s\n", (*iter)->ID, (*iter)->name);
        if (*iter == glv_curr) glv_curr = NULL;
        if (*iter == glv_temp) glv_temp = NULL;
        delete (*iter);
        iter++;
    }
}
void CtrlLouvain::CleanAll() {
    CleanList();
    if (glv_curr != NULL) delete (glv_curr);
    if (glv_temp != NULL) delete (glv_temp);
    glv_curr = glv_temp = NULL;
}
CtrlLouvain::~CtrlLouvain() {
    CleanList();
    printf("\033[1;37;40mINFO\033[0m: CtrlLouvain de-structed \n");
}
list<GLV*>::iterator CtrlLouvain::IsInList(int id) {
    list<GLV*>::iterator iter = glv_list.begin(); // std::find(glv_list.begin(),glv_list.end(), p_push);
    while (iter != glv_list.end()) {
        if ((*iter)->ID == id) return iter;
        iter++;
    }
    return iter;
}
GLV* CtrlLouvain::IsInList_glv(int id) {
    list<GLV*>::iterator iter = glv_list.begin(); // std::find(glv_list.begin(),glv_list.end(), p_push);
    while (iter != glv_list.end()) {
        if ((*iter)->ID == id) return *iter;
        iter++;
    }
    return NULL;
}
bool CtrlLouvain::IsInList(GLV* it) {
    list<GLV*>::iterator iter = glv_list.begin(); // std::find(glv_list.begin(),glv_list.end(), p_push);
    while (iter != glv_list.end()) {
        if ((*iter) == it) return true;
        iter++;
    }
    return false;
}
GLV* CtrlLouvain::find(int id) {
    GLV* ret = IsInList_glv(id);
    if (ret != NULL) return ret;
    if (glv_curr->ID == id) return glv_curr;
    if (glv_temp->ID == id) return glv_temp;
    return NULL;
}
bool CtrlLouvain::SafeCleanTemp() {
    if (glv_temp == NULL) return false;
    if (IsInList(glv_temp))
        return false;
    else
        glv_temp->CleanCurrentG();
    return true;
}
#define CHECKCURR                                                                               \
    if (glv_curr == 0) {                                                                        \
        printf("\033[1;31;40mERROR\033[0m: glv_curr should not be NULL for running Louvain\n"); \
        return -1;                                                                              \
    }
int CtrlLouvain::pushList(GLV* p_push) {
    list<GLV*>::iterator iter = glv_list.begin(); // std::find(glv_list.begin(),glv_list.end(), p_push);
    while (iter != glv_list.end()) {
        if ((*iter)->ID == p_push->ID) break;
        iter++;
    }
    if (iter != glv_list.end()) {
        printf("\033[1;31;40mERROR\033[0m: The GLV with ID:%d already exist in list\n", (*iter)->ID);
        p_push->printSimple();
        return -1;
    }
    glv_list.push_back(p_push);
    cnt_list++;
    return 0;
}
int CtrlLouvain::exe_LV_PUSH() {
    // push
    // push curr|tmp|temp
    GLV* p_push = glv_curr;
    if (mycmd.argc == 1)
        p_push = glv_temp;
    else if (mycmd.argc == 2 && (strcmp("temp", mycmd.argv[1]) == 0 || strcmp("tmp", mycmd.argv[1]) == 0))
        p_push = glv_temp;

    if (p_push == 0) {
        printf("\033[1;31;40mERROR\033[0m: Object is empty\n");
        return -1;
    };

    return pushList(p_push);
    // return 0;
}
int CtrlLouvain::exe_LV_POP() {
    // pop id
    // pop id cur|tmp|temp
    if (mycmd.argc == 1) {
        printf("\033[1;31;40mERROR\033[0m: Wrong parameter! Please using :pop <ID> [curr|tmp|temp]\n");
        return -1;
    }
    int id = atoi(mycmd.argv[1]);
    GLV** p_pop = &glv_curr;
    if (mycmd.argc == 3) {
        if ((strcmp("temp", mycmd.argv[2]) == 0 || strcmp("tmp", mycmd.argv[2]) == 0)) {
            p_pop = &glv_temp;
            if (IsInList(glv_temp)) delete (glv_temp);
        }
    }
    list<GLV*>::iterator iter = glv_list.begin(); // std::find(glv_list.begin(),glv_list.end(), p_push);
    while (iter != glv_list.end()) {
        if ((*iter)->ID == id) break;
        iter++;
    }
    if (iter == glv_list.end()) {
        printf("\033[1;31;40mERROR\033[0m: There is no GLV with ID:%d in list\n", id);
        return -1;
    }
    *p_pop = (*iter);
    return 0;
}
void GraphCat_general(graphNew* g_cat, graphNew* g1, graphNew* g2, long num_add) {
    long nv1 = g1->numVertices;
    long nv2 = g2->numVertices;
    long nv3 = nv1 + nv2;
    long ne1 = g1->numEdges;
    long ne2 = g2->numEdges;
    long ne3 = ne1 + ne2 + num_add;

    edge* edgeList = (edge*)malloc(2 * ne3 * sizeof(edge));

    long cnt_1 = 0;
    for (int i = 0; i < g1->edgeListPtrs[nv1]; i++) {
        if (g1->edgeList[i].head > g1->edgeList[i].tail) continue;
        edgeList[cnt_1].head = g1->edgeList[i].head;
        edgeList[cnt_1].tail = g1->edgeList[i].tail;
        edgeList[cnt_1].weight = g1->edgeList[i].weight;
        cnt_1++;
    }
    long cnt_2 = 0;
    long off_v = cnt_1;
    for (int i = 0; i < g2->edgeListPtrs[nv2]; i++) {
        if (g2->edgeList[i].head > g2->edgeList[i].tail) continue;
        edgeList[cnt_2 + off_v].head = g2->edgeList[i].head + nv1;
        edgeList[cnt_2 + off_v].tail = g2->edgeList[i].tail + nv1;
        edgeList[cnt_2 + off_v].weight = g2->edgeList[i].weight;
        cnt_2++;
    }
    off_v += cnt_2;
    for (int i = 0; i < num_add; i++) {
        long h = rand() % nv1;
        long t = rand() % nv2 + nv1;
        edgeList[i + off_v].head = h;
        edgeList[i + off_v].tail = t;
        edgeList[i + off_v].weight = 1.0;
    }
    GetGFromEdge(g_cat, edgeList, nv3, ne3);
    return;
}
void GraphCat_general(graphNew* g_cat, graphNew* g1, graphNew* g2, double rate) {
    long nv1 = g1->numVertices;
    long nv2 = g2->numVertices;
    long num_add = (nv1 + nv2) * rate;
    GraphCat_general(g_cat, g1, g2, num_add);
}
void GraphCat_general(graphNew* g_cat, graphNew* g1, graphNew* g2) {
    long nv1 = g1->numVertices;
    long nv2 = g2->numVertices;
    long nv3 = nv1 + nv2;
    long ne1 = g1->numEdges;
    long ne2 = g2->numEdges;
    long ne3 = ne1 + ne2;

    g_cat->edgeListPtrs = (long*)malloc((nv3 + 1) * sizeof(long));
    assert(g_cat->edgeListPtrs);
    g_cat->edgeList = (edge*)malloc((g1->edgeListPtrs[nv1] + g2->edgeListPtrs[nv2]) * sizeof(edge));
    assert(g_cat->edgeList);

    g_cat->numVertices = nv1 + nv2;
    g_cat->numEdges = ne1 + ne2;
    g_cat->sVertices = 0;

    for (int i = 0; i < nv1 + 1; i++) g_cat->edgeListPtrs[i] = g1->edgeListPtrs[i];
    for (int i = 0; i < nv2 + 1; i++) g_cat->edgeListPtrs[i + nv1] = g2->edgeListPtrs[i] + g1->edgeListPtrs[nv1];
    for (int i = 0; i < g1->edgeListPtrs[nv1]; i++) {
        g_cat->edgeList[i].head = g1->edgeList[i].head;
        g_cat->edgeList[i].tail = g1->edgeList[i].tail;
        g_cat->edgeList[i].weight = g1->edgeList[i].weight;
    }
    long off_v = g1->edgeListPtrs[nv1];
    for (int i = 0; i < g2->edgeListPtrs[nv2]; i++) {
        g_cat->edgeList[i + off_v].head = g2->edgeList[i].head + nv1;
        g_cat->edgeList[i + off_v].tail = g2->edgeList[i].tail + nv1;
        g_cat->edgeList[i + off_v].weight = g2->edgeList[i].weight;
    }

    return;
}
GLV* GraphCat_general(GLV* glv_cat, GLV* glv1, GLV* glv2, double rate) {
    graphNew* g_cat = (graphNew*)malloc(sizeof(graphNew));
    GraphCat_general(g_cat, glv1->G, glv2->G, rate);
    glv_cat->SetByOhterG(g_cat);
    return glv_cat;
}
GLV* GraphCat_general(GLV* glv_cat, GLV* glv1, GLV* glv2) {
    graphNew* g_cat = (graphNew*)malloc(sizeof(graphNew));
    GraphCat_general(g_cat, glv1->G, glv2->G);
    glv_cat->SetByOhterG(g_cat);
    return glv_cat;
}
int CtrlLouvain::exe_LV_CAT() {
    // del id
    if (mycmd.argc < 3) {
        printf("\033[1;31;40mERROR\033[0m: Wrong parameter! Please using :cat [<id>] [<id>] <ID>\n");
        return -1;
    }
    GLV* p_1 = NULL;
    GLV* p_2 = NULL;

    int id1 = atoi(mycmd.argv[1]);
    int id2 = atoi(mycmd.argv[2]);

    list<GLV*>::iterator iter1 = IsInList(id1);
    if (iter1 != glv_list.end()) {
        p_1 = (*iter1);
    } else {
        printf("\033[1;31;40mERROR\033[0m: No GLV object id=%d stored in glv_list \n", id1);
        return -1;
    }
    list<GLV*>::iterator iter2 = IsInList(id2);
    if (iter2 != glv_list.end()) {
        p_2 = (*iter2);
    } else {
        printf("\033[1;31;40mERROR\033[0m: No GLV object id=%d stored in glv_list \n", id2);
        return -1;
    }
    int id_rate = mycmd.cmd_findPara("-r");
    double rate = 0;
    if (id_rate != -1 && mycmd.argc > id_rate) rate = atof(mycmd.argv[id_rate + 1]);
    this->SafeCleanTemp();
    GLV* p_cat = new GLV(id_glv);
    this->glv_temp = GraphCat_general(p_cat, p_1, p_2, rate);
    this->glv_temp->SetName_cat(p_1->ID, p_2->ID);
    return 0;
}
int CtrlLouvain::exe_LV_DEL() {
    // del id
    if (mycmd.argc == 1) {
        printf("\033[1;31;40mERROR\033[0m: Wrong parameter! Please using :del <ID>\n");
        return -1;
    }
    int id = atoi(mycmd.argv[1]);
    GLV* p_del = NULL;
    if (glv_curr && (glv_curr->ID == id)) {
        p_del = glv_curr;
        glv_curr = NULL;
    }
    if (glv_temp && (glv_temp->ID == id)) {
        p_del = glv_temp;
        glv_temp = NULL;
    }
    list<GLV*>::iterator iter = IsInList(id);
    if (iter != glv_list.end()) {
        p_del = (*iter);
        glv_list.erase(iter);
        cnt_list--;
    }
    if (p_del == NULL) {
        printf("\033[1;31;40mERROR\033[0m: No %d ID in system\n", id);
        return -1;
    }
    printf("\033[1;37;40mINFO\033[0m: Found %d ID in system\n", id);
    printf("Deleting... \n");
    p_del->printSimple();
    delete (p_del);
    printf("Delete Done\n");
    return 0;
}
int SaveG_general(graphNew* g, char* fileName) {
    FILE* file = fopen(fileName, "w");
    if (file == NULL) {
        printf("\033[1;31;40mERROR\033[0m: Cannot open the batch file: %s\n", fileName);
        return -1;
    }
    long nv = g->numVertices;
    long ne = g->numEdges;
    map<long, long> map_self;
    map<long, long>::iterator itr;
    int cnt = 0;
    int cnt_self = 0;
    fprintf(file, "*Vertices %d\n", nv);
    fprintf(file, "*Edges %d\n", ne);
    for (int i = 0; i < g->edgeListPtrs[nv]; i++) {
        long h = g->edgeList[i].head + 1;
        long t = g->edgeList[i].tail + 1;
        double w = g->edgeList[i].weight;
        if (h > t) continue;
        if (h == t) {
            itr = map_self.find(h);
            if (itr != map_self.end()) {
                continue;
            }
            map_self[h] = cnt_self++;
        }
        fprintf(file, "%d %d %f\n", h, t, w);
        cnt++;
    }

    fclose(file);
    printf("\033[1;37;40mINFO\033[0m: -f 3 file samed name as %s\n", fileName);
    printf("\033[1;37;40mINFO\033[0m: -f 3 file samed NV is  %d\n", nv);
    printf("\033[1;37;40mINFO\033[0m: -f 3 file samed |NE| is  %d and self edge is %d\n", ne);
    printf("\033[1;37;40mINFO\033[0m: -f 3 file samed undirection edge has  %d \n", cnt);
    printf("\033[1;37;40mINFO\033[0m: -f 3 file samed sef-loop edge has %d\n", cnt_self);
    return 0;
}
int CtrlLouvain::exe_LV_SAVE() {
    // del id
    GLV* p_glv = NULL;
    char* filename;
    int id;
    if (mycmd.argc == 1) {
        p_glv = glv_curr;
    } else if (mycmd.argc == 2) {
        printf("\033[1;31;40mERROR\033[0m: Wrong parameter! Please using :save <ID>\n");
        return -1;
    } else if (mycmd.argc == 3) {
        if (mycmd.cmd_findPara("-d") != -1) {
            id = atoi(mycmd.argv[2]);
            p_glv = find(id);
        }
        if (mycmd.cmd_findPara("-f") != -1) {
            filename = mycmd.argv[2];
            p_glv = glv_curr;
        }
    }
    if (p_glv == NULL) {
        printf("\033[1;31;40mERROR\033[0m: Wrong parameter! Please using :save <ID>\n");
        return -1;
    }
    filename = p_glv->name;
    if (SaveG_general(p_glv->G, filename) == -1) {
        printf("\033[1;31;40mERROR\033[0m: When running SaveG_general()\n");
        return -1;
    }
    printf("\033[1;37;40mINFO\033[0m: Graph saved as %s in disk\n", filename);
    return 0;
}
int CtrlLouvain::exe_LV_DEMO() {
    mycmd.cmd_GetCmd("loadg ../data/as-Skitter-wt.mtx");
    ExeCmd();
    /*	mycmd.cmd_GetCmd("par -num 2 -prun 1");
            ExeCmd();
            mycmd.cmd_GetCmd("par -lv");
            ExeCmd();
            mycmd.cmd_GetCmd("par -pre");
            ExeCmd();
            mycmd.cmd_GetCmd("par -merge");
            ExeCmd();
            mycmd.cmd_GetCmd("par -final");
            ExeCmd();
            mycmd.cmd_GetCmd("pl");
                    ExeCmd();*/
    // mycmd.cmd_GetCmd("loadg data/part1.bat");
    return 0;
}
int CtrlLouvain::exe_batch(char* fileName) {
    char nameBatch[1024];
    strcpy(nameBatch,
           fileName); // because cmd.argc will be overloaded which is source of fileName, copy it into nameBatch
    FILE* file = fopen(nameBatch, "r");
    if (file == NULL) {
        printf("\033[1;31;40mERROR\033[0m: Cannot open the batch file: %s\n", fileName);
        return -1;
    }

    int cnt_exe = 0;
    do {
        char line[1024] = "";
        fgets(line, 1024, file);
        if (line[0] == '#' || line[0] == '\0') // This line is a comment line
            continue;
        mycmd.cmd_GetCmd(line);
        if (mycmd.cmd_last == LVHISTORY) continue;
        if (mycmd.cmd_last == LVLOADBAT) {
            if (mycmd.argc > 1) {
                if (strcmp(nameBatch, mycmd.argv[1]) == 0) {
                    printf("\033[1;31;40mERROR\033[0m: Dead-loop for opening batch file detected!!! %s\n", nameBatch);
                    printf("\033[1;31;40mERROR\033[0m: Batch file can not open itself !!!\n");
                    fclose(file);
                    return -1;
                }
            }
        }
        if (this->ExeCmd() == -1) {
            printf("\033[1;31;40mERROR\033[0m: Cannot execute the %dth command %s\n", cnt_exe + 1, nameBatch);
            fclose(file);
            return -1;
        }
        cnt_exe++;
    } while (!feof(file));
    printf("\033[1;37;40mINFO\033[0m: Done reading execute batch file %s\n", nameBatch);
    if (cnt_exe == 1)
        printf("\033[1;37;40mINFO\033[0m: One command is executed.\n");
    else
        printf("\033[1;37;40mINFO\033[0m: Total %d commands are executed.\n", cnt_exe);
    fclose(file);
    return 0;
}
int CtrlLouvain::exe_LV_LOADBAT() {
    int argc = mycmd.argc;
    if (argc != 2) {
        printf("\033[1;31;40mERROR\033[0m: mycmd.argc = %d, it should be '2' or more \n", mycmd.argc);
        return -1;
    } else {
        return exe_batch(mycmd.argv[1]);
    }
    return 0;
}
int CtrlLouvain::exe_LV_LOADG() {
    int argc = mycmd.argc;
    int i_nm_plv = mycmd.cmd_findPara("-name");
    int i_nm_file = i_nm_plv == -1 ? 1 : 3;
    if (argc == 1 || argc == 3) {
        printf("\033[1;31;40mERROR\033[0m: mycmd.argc = %d, it should be '2' or more \n", mycmd.argc);
        return -1;
    } else if (argc == 2 && i_nm_file == 1) {
        glv_curr = new GLV(id_glv);
        this->glv_curr->InitByFile(mycmd.argv[1]);
        glv_curr->SetName_loadg(glv_curr->ID, mycmd.argv[1]);
    } else {
        glv_curr = new GLV(id_glv);
        this->glv_curr->InitByFile(mycmd.argv[i_nm_file]);
        this->glv_curr->SetName(mycmd.argv[i_nm_plv + 1]);
    }
    glv_list.push_back(glv_curr);
    cnt_list++;
    return 0;
}
int CtrlLouvain::exe_LV_MODUG() {
    if (mycmd.cmd_findPara("--help") != -1) return 1; // exe_LV_MODUG();
    bool isUsingMethod2 = mycmd.cmd_findPara("-m2") != -1;
    GLV* pglv = findByCmd();
    if (pglv == NULL) {
        printf("\033[1;31;40mERROR\033[0m:Object is NULL or id can not be found\n");
        return -1;
    }
    FeatureLV f1;
    if (!isUsingMethod2)
        f1.ComputeQ(pglv);
    else
        f1.ComputeQ2(pglv);
    f1.PrintFeature();
    return 0;
}
int CtrlLouvain::exe_LV_QUIT() {
    return 0;
}
int CtrlLouvain::exe_LV_STATUS() {
    ShowPara();
    return 0;
}
int CtrlLouvain::exe_LV_SETM() {
    if (mycmd.argc != 2) {
        printf("\033[1;31;40mERROR\033[0m: exe_LV_SETM wrong number of parameter!\n");
        return -1;
    }
    this->minGraphSize = atoi(mycmd.argv[1]);
    return 0;
}
int CtrlLouvain::exe_LV_SETTH() {
    if (mycmd.argc != 2) {
        printf("\033[1;31;40mERROR\033[0m: exe_LV_SETTH wrong number of parameter!\n");
        return -1;
    }
    this->threshold = atof(mycmd.argv[1]);
    return 0;
}
int CtrlLouvain::exe_LV_SETTHC() {
    if (mycmd.argc != 2) {
        printf("\033[1;31;40mERROR\033[0m: exe_LV_SETTHC wrong number of parameter!\n");
        return -1;
    }
    this->C_threshold = atof(mycmd.argv[1]);
    return 0;
}
int CtrlLouvain::man_LV_PG() {
    printf("Help \"pg\" used for printing graphNew content.\n");
    printf("pg :default glv is glv_curr, print all content\n");
    printf("pg [<start>, <end>]: default glv is glv_curr, print content from start to end\n");
    printf("pg [<start>, <end>] [-id <id>]\n");
    printf("pg [-id <id>] [<start>, <end>]\n");
    return 1;
}
void printG(graphNew* G, long* C, long* M, long star, long end, bool isCid, bool isDir, ParLV* p_par, int idx);
int CtrlLouvain::exe_LV_PG() { // 1)cmd 2)cmd [-id <id>] [<start> <end>] 3)cmd [<start> <end>] [-id <id>] [-c]
    CHECKCURR;
    if (mycmd.cmd_findPara("--help") != -1) return man_LV_PG();
    GLV* pglv = glv_curr;
    long p1 = 0;
    long p2 = pglv->NV;
    int id;
    int i_id = mycmd.cmd_findPara("-id");
    int i_star;

    bool isCid = mycmd.cmd_findPara("-c") != -1 || mycmd.cmd_findPara("-cid") != -1;
    bool isDir = mycmd.cmd_findPara("-d") != -1 || mycmd.cmd_findPara("-dir") != -1;
    bool isPar = mycmd.cmd_findPara("-par") != -1 && (parlv.st_ParLved);

    if (i_id != -1 && i_id < mycmd.argc - 1) {
        id = atoi(mycmd.argv[i_id + 1]);
        pglv = find(id);
        if (pglv == NULL) {
            printf("\033[1;31;40mERROR\033[0m: Wrong index for glv id %d  \n", id);
            return -1;
        }
        if (i_id == 3) {
            p1 = atoi(mycmd.argv[1]);
            p2 = atoi(mycmd.argv[2]);
        }
    } else if (mycmd.argc >= 3) {
        p1 = atoi(mycmd.argv[1]);
        p2 = atoi(mycmd.argv[2]);
    }
    //-f
    int idx_f = mycmd.cmd_findPara("-f");
    if (idx_f == -1) {
        printf("Graph: %s Local V = %d  \t loval |E| = %d  \t  ", pglv->name, pglv->NVl, pglv->NElg);
        if (!isPar)
            printG(pglv->G, pglv->C, pglv->M, p1, p2, isCid, isDir);
        else
            printG(pglv->G, pglv->C, pglv->M, p1, p2, isCid, isDir, &parlv, parlv.FindParIdxByID(id));
        return 0;
    }
    char filename[256];
    sprintf(filename, "%s_%d_to%d", (mycmd.argc > idx_f) ? mycmd.argv[idx_f + 1] : pglv->name, p1, p2);
    printG(filename, pglv->G, pglv->C, pglv->M, p1, p2);
    return 0;
}
int CtrlLouvain::exe_LV_MERGE() { // 1)cmd 2)cmd [-id <id>]
    GLV* pglv = findByCmd();
    if (pglv == NULL) {
        printf("\033[1;31;40mERROR\033[0m:Object is NULL or id can not be found\n");
        return -1;
    }
    SafeCleanTemp();
    graphNew* Gnew = (graphNew*)malloc(sizeof(graphNew));
    assert(Gnew != 0);
    double TimeBuildingPhase = buildNextLevelGraphOpt(pglv->G, Gnew, pglv->C, pglv->NV, numThreads);
    printG(Gnew, pglv->C);
    FreeG(Gnew);
    // glv_temp->SetByOhterG(Gnew);
    return 0;
}
GLV* CtrlLouvain::findByCmd() { // classic for 1)cmd 2)cmd tmp|temp|curr 3)cmd -id <id>
    GLV* pglv = glv_curr;
    if (mycmd.argc == 1) return pglv;
    int id;
    int i_id = mycmd.cmd_findPara("-id");
    if (i_id != -1) {
        if (i_id < mycmd.argc - 1) {
            id = atoi(mycmd.argv[i_id + 1]);
            pglv = find(id);
        } else {
            printf("\033[1;31;40mERROR\033[0m: Too few parameter for -id\n");
            return NULL;
        }
    } else if (mycmd.argc == 2) {
        if ((strcmp("temp", mycmd.argv[1]) == 0 || strcmp("tmp", mycmd.argv[1]) == 0))
            pglv = glv_temp;
        else if (strcmp("curr", mycmd.argv[1]) == 0)
            pglv = glv_curr;
        else {
            printf("\033[1;31;40mERROR\033[0m: Wrong parameter %s \n", mycmd.argv[1]);
            return NULL;
        }
    } else {
        printf("\033[1;31;40mERROR\033[0m: Wrong number of parameter %d \n", mycmd.argc);
        return NULL;
    }
    return pglv;
}

GLV* CtrlLouvain::findByCmd(int& i_p2) { // classic for 1)cmd <p2> 2)cmd tmp|temp|curr <p2> 3)cmd -id <id> <p2>
    GLV* pglv = glv_curr;
    i_p2 = 1;
    int id;
    int i_id = mycmd.cmd_findPara("-id");
    if (i_id != -1) {
        if (i_id < mycmd.argc - 1) {
            id = atoi(mycmd.argv[i_id + 1]);
            pglv = find(id);
        } else {
            printf("\033[1;31;40mERROR\033[0m: Too few parameter for -id\n");
            return NULL;
        }
        if (mycmd.argc != 4) {
            printf("\033[1;31;40mERROR\033[0m: Too few parameter for -id\n");
            return NULL;
        }
        if (i_id == 1)
            i_p2 = 3;
        else
            i_p2 = 1;
    } else if (mycmd.argc == 3) {
        if ((strcmp("temp", mycmd.argv[1]) == 0 || strcmp("tmp", mycmd.argv[1]) == 0))
            pglv = glv_temp;
        else if (strcmp("curr", mycmd.argv[1]) == 0)
            pglv = glv_curr;
        else {
            printf("\033[1;31;40mERROR\033[0m: Wrong parameter %s \n", mycmd.argv[1]);
            return NULL;
        }
        i_p2 = 2;
    } else if (mycmd.argc != 2) {
        printf("\033[1;31;40mERROR\033[0m: Too more parameter for -id\n");
        return NULL;
    }
    return pglv;
}

int CtrlLouvain::exe_LV_RSTC() {                      // classic for 1)cmd 2)cmd tmp|temp 3)cmd -id <id>
    if (mycmd.cmd_findPara("--help") != -1) return 1; // man_LV_PG();
    GLV* pglv = findByCmd();
    if (pglv == NULL) {
        printf("CTRL::exe_LV_RSTC::Object is NULL or id can not be found\n");
        return -1;
    }
    pglv->ResetC();
    return 0;
}
int CtrlLouvain::exe_LV_RSTCL() {                     // classic for 1)cmd 2)cmd tmp|temp 3)cmd -id <id>
    if (mycmd.cmd_findPara("--help") != -1) return 1; // man_LV_PG();
    GLV* pglv = findByCmd();
    if (pglv == NULL) {
        printf("CTRL::exe_LV_RSTCL::Object is NULL or id can not be found\n");
        return -1;
    }
    pglv->ResetColor();
    return 0;
}
int CtrlLouvain::exe_LV_INFOG() {
    return 0;
}
int CtrlLouvain::exe_LV_UNIQ() { // 1)cmd <start> <end>
    CHECKCURR;
    long cnt;
    SttGPar stt;
    edge* pedge;
    long* M_v;
    // pedge = (edge*)malloc(sizeof(edge) * (glv_curr->G->numEdges*2));
    // M_v   = (long*)malloc(sizeof(long) * (glv_curr->G->numVertices));
    if (mycmd.argc == 1) {
        stt.CountV(glv_curr->G); //, pedge, M_v);
    } else {
        if (mycmd.argc == 3) {
            long p1 = atoi(mycmd.argv[1]);
            long p2 = atoi(mycmd.argv[2]);
            if (p1 < p2) stt.CountV(glv_curr->G, p1, p2); //, pedge, M_v );
                                                          // stt.CountVPruning(glv_curr->G, p1,p2);//, pedge, M_v );
            else
                stt.CountV(glv_curr->G, p2, p1); //, pedge, M_v );
        } else {
            printf("\033[1;37;40mINFO\033[0m::exe_LV_UNIQ incorrect parameter = %d, it should be uniq <star> <end> \n",
                   mycmd.argc - 1);
            return -1;
        }
    }
    stt.PrintStt();
    // free(pedge);
    // free(M_v);
    return 0;
}

int CtrlLouvain::exe_LV_TEST() {
    return 0;
}
void CtrlLouvain::print_parlv() {
    if (parlv.st_Partitioned == false) {
        printf("\033[1;37;40mINFO\033[0m: Partition handle is \033[1;31;40mNOT LOADED\033[0m \n");
        return;
    }
    printf(
        "=========================================================[ \033[1;35;40mPar_SRC BEGIN\033[0m "
        "]======================================================================================\n");
    int cnt = 0;
    for (int p = 0; p < parlv.num_par; p++) {
        printf("| \033[1;35;40mSrc\033[0m:%2d off:%4d ", cnt++, parlv.off_src[p]);
        parlv.par_src[p]->printSimple();
    }
    printf(
        "=============================================================================================================="
        "======================================================================\n");
    if (parlv.st_ParLved == false) {
        printf("\033[1;37;40mINFO\033[0m: Partition handle is \033[1;31;40mNOT Louvained\033[0m \n");
        return;
    }
    for (int p = 0; p < parlv.num_par; p++) {
        printf("| \033[1;35;40mLv \033[0m:%2d off:%4d ", cnt++, parlv.off_lved[p]);
        parlv.par_lved[p]->printSimple();
    }
    printf(
        "=========================================================[ \033[1;35;40mPar_lved  END\033[0m "
        "]======================================================================================\n");
}
void CtrlLouvain::print_glv_list() {
    list<GLV*>::iterator iter = glv_list.begin();
    printf(
        "=========================================================[ \033[1;35;40mLIST BEGIN\033[0m "
        "]======================================================================================\n");
    int cnt = 0;
    while (iter != glv_list.end()) {
        printf("| \033[1;35;40mList\033[0m:%4d    ", cnt++); //"\033[1;34;40mERROR\033[0m:  \033[0m
        (*iter)->printSimple();
        iter++;
    }
    printf(
        "=========================================================[ \033[1;35;40mLIST   END\033[0m "
        "]======================================================================================\n");
}
int CtrlLouvain::exe_LV_PL() {
    if (mycmd.cmd_findPara("-par") == -1)
        print_glv_list();
    else
        print_parlv();
    return 0;
}

GLV* CtrlLouvain::CtrlParOne(GLV* src, long start, long end, bool isPrun, int th_prun) {
    // SafeCleanTemp();
    return par_general(src, id_glv, start, end, isPrun, th_prun);
}

double CtrlLouvain::CtrlPar(GLV* src, int num_par, bool isPrun, int th_prun) {
    double time_fun = omp_get_wtime();
    long vsize = src->NV / num_par;
    long start = 0;
    long end = start + vsize;

    parlv.Init(flowMode, src, num_par,
               this->num_dev); // should never release resource and object who pointed; Just work as a handle
    for (int p = 0; p < num_par; p++) {
        double time_par = omp_get_wtime();
        GLV* tmp = par_general(src, &(parlv.stt[p]), id_glv, start, end, isPrun, th_prun);
        parlv.timesPar.timePar[p] = omp_get_wtime() - time_par;
        parlv.par_src[p] = tmp;
        // parlv.off_src[p+1]=end;
        pushList(tmp);
        start = end;
        end = (p == num_par - 2) ? src->NV : start + vsize;
    }
    return omp_get_wtime() - time_fun;
}

int CtrlLouvain::exe_LV_PAR() {
    // par [curr] star end [temp]
    CHECKCURR;
    assert(glv_curr->G);
    int idx_prun = mycmd.cmd_findPara("-prun");
    int th_prun = 1;

    if (idx_prun > 0 && mycmd.argc > idx_prun) {
        th_prun = atoi(mycmd.argv[idx_prun + 1]);
        if (th_prun > MAXGHOST_PRUN || th_prun < 1) {
            printf(
                "\033[1;31;40mERROR\033[0m: Wrong value of partition parameter for -prun which should be in range of 1 "
                "~%d!\n",
                MAXGHOST_PRUN);
            return -1;
        }
    }
    bool isPrun = idx_prun > 0;
    int idx_num = mycmd.cmd_findPara("-num");
    int num_par = 1;
    if (idx_num > 0 && mycmd.argc > idx_num) {
        num_par = atoi(mycmd.argv[idx_num + 1]);
        if (num_par > MAX_PARTITION || num_par < 1) {
            printf(
                "\033[1;31;40mERROR\033[0m: Wrong value of partition parameter for -num which should be in range of 1 "
                "~%d!\n",
                MAX_PARTITION);
            return -1;
        }
    }
    bool isNumber = idx_num > 0;

    bool isAllParFlow = mycmd.cmd_findPara("-all") != -1;

    if (mycmd.argc >= 3) {
        if (!isNumber) {
            long start = atoi(mycmd.argv[1]);
            long end = atoi(mycmd.argv[2]);
            if (start >= end) {
                printf("\033[1;31;40mERROR\033[0m: start:%d > end%d\n", start, end);
                return -1;
            } else if (end > glv_curr->NV) {
                printf("\033[1;31;40mERROR\033[0m: end:%d > NV:%d\n", end, glv_curr->NV);
                return -1;
            }
            SafeCleanTemp();
            glv_temp = CtrlParOne(glv_curr, start, end, isPrun, th_prun);
        } // if(!isNumber)
        else if (!isAllParFlow) {
            if (num_par >= MAX_PARTITION) {
                printf("\033[1;31;40mERROR\033[0m: wrong number of partition %d which should be small than %d!\n",
                       num_par, MAX_PARTITION);
                return -1;
            }
            SafeCleanTemp();
            if (mycmd.cmd_findPara("-lv") == -1)
                parlv.timesPar.timePar_all = CtrlPar(glv_curr, num_par, isPrun, th_prun);
            else
                ParLV_thread(glv_curr, num_par, isPrun, th_prun);
            printf("\033[1;37;40mINFO\033[0m:Total time for partition : %lf\n", parlv.timesPar.timePar_all);
        } else { // do all flow
            SafeCleanTemp();
            glv_temp =
                LouvainGLV_general_par(flowMode, glv_curr, num_par, num_dev, isPrun, th_prun, xclbinPath, numThreads,
                                       id_glv, minGraphSize, threshold, C_threshold, this->isParallel, this->numPhase);
        }
    } else if (mycmd.cmd_findPara("-lv") != -1) {
        if (parlv.st_ParLved == true) {
            printf("\033[1;37;40mINFO\033[0m: Louvains for sub-graphNew have already been done!!!\n");
            return -1;
        }
        if (CtrlPar_Louvain() == -1) return -1;
        printf("\033[1;37;40mINFO\033[0m: Total time for partition Louvains : %lf\n", parlv.timesPar.timeLv_all);
    } else if (mycmd.cmd_findPara("-pre") != -1 || mycmd.cmd_findPara("-premerge") != -1) {
        if (parlv.st_PreMerged == true) {
            printf("\033[1;37;40mINFO\033[0m: Pre-merging has already been done!!!\n");
            return -1;
        }
        parlv.timesPar.timePre = omp_get_wtime();
        parlv.PreMerge();
        parlv.timesPar.timePre = omp_get_wtime() - parlv.timesPar.timePre;
        printf("\033[1;37;40mINFO\033[0m: Total time for partition pre-Merge  : %lf\n", parlv.timesPar.timePre);
    } else if (mycmd.cmd_findPara("-merge") != -1) {
        if (parlv.st_Merged == true) {
            printf("\033[1;37;40mINFO\033[0m: Merging has already been done!!!\n");
            return -1;
        }
        if (parlv.st_PreMerged == false) {
            printf("\033[1;37;40mINFO\033[0m: Doing Pre-merging firstly\n");
            parlv.timesPar.timePre = omp_get_wtime();
            parlv.PreMerge();
            parlv.timesPar.timePre = omp_get_wtime() - parlv.timesPar.timePre;
            printf("\033[1;37;40mINFO\033[0m: Total time for partition pre-Merge : %lf\n", parlv.timesPar.timePre);
        }
        SafeCleanTemp();
        parlv.timesPar.timeMerge = omp_get_wtime();
        glv_temp = parlv.MergingPar2(id_glv);
        parlv.timesPar.timeMerge = omp_get_wtime() - parlv.timesPar.timeMerge;
        printf("\033[1;37;40mINFO\033[0m: Total time for partition Merge : %lf\n", parlv.timesPar.timeMerge);
        pushList(glv_temp);
    } else if (mycmd.cmd_findPara("-final") != -1 || mycmd.cmd_findPara("-flv") != -1) {
        if (parlv.st_Merged == false) {
            printf("\033[1;31;40mERROR\033[0m:Please do merging and other previous processing\n");
            return -1;
        }
        SafeCleanTemp();
        parlv.timesPar.timeFinal = omp_get_wtime();
        glv_temp = parlv.FinalLouvain(xclbinPath, numThreads, id_glv, minGraphSize, threshold, C_threshold,
                                      this->isParallel, this->numPhase);
        parlv.timesPar.timeFinal = omp_get_wtime() - parlv.timesPar.timeFinal;
        parlv.UpdateTimeAll();
        parlv.PrintTime();
        pushList(glv_temp);
        // To update original graphNew with new number of community NC and modularity Q
        parlv.plv_src->NC = glv_temp->NC;
        parlv.plv_src->PushFeature(0, 0, 0.0, true);
    } else if (mycmd.cmd_findPara("-info") != -1 || mycmd.cmd_findPara("-print") != -1) {
        if (parlv.st_Partitioned == false) {
            printf("\033[1;31;40mERROR\033[0m: Partition Handle are not loaded\n");
            return -1;
        }
        parlv.PrintSelf();
    } else {
        printf("\033[1;31;40mERROR\033[0m: Wrong number of parameter!\n");
        printf("Command for 'par': %s\n", cmdlist_intro[LVPAR]);
        return -1;
    }

    return 0;
}

int CtrlLouvain::exe_LV_LV() {
    CHECKCURR;
    assert(glv_curr->G);

    int idx_phase = mycmd.cmd_findPara("-phase");
    int n_phase = 100;
    if (idx_phase > 0 && mycmd.argc > idx_phase) {
        n_phase = atoi(mycmd.argv[idx_phase + 1]);
        if (n_phase < 1) {
            printf("\033[1;31;40mERROR\033[0m: Wrong value of n_phase. It should be no smaller than 1\n");
            return -1;
        } else {
            this->numPhase = n_phase;
            printf("| \033[1;32;40m New numPhase \033[0m   : %d\n", numPhase); //= 0.0001; //default 0.0001
        }
    }

    if (mycmd.cmd_findPara("-parallel") != -1 || mycmd.cmd_findPara("-p") != -1) this->isParallel = true;
    if (mycmd.cmd_findPara("-noparallel") != -1 || mycmd.cmd_findPara("-nop") != -1 ||
        mycmd.cmd_findPara("-noParallel") != -1)
        this->isParallel = false;

    bool hasGhost = false;
    if (mycmd.cmd_findPara("-gh") != -1 || mycmd.cmd_findPara("-ghost") != -1) hasGhost = true;

    if (mycmd.argc == 1 || idx_phase != -1) {
        SafeCleanTemp();
        glv_temp = LouvainGLV_general(hasGhost, this->flowMode, 0, glv_curr, xclbinPath, numThreads, id_glv,
                                      minGraphSize, threshold, C_threshold, this->isParallel, this->numPhase);
        return 0;
    } else if (mycmd.cmd_findPara("-history") != -1) {
        glv_curr->printFeature();
        return 0;
    } else if (mycmd.cmd_findPara("-par") != -1) {
        return CtrlPar_Louvain();
    } else if (mycmd.cmd_findPara("-merge") != -1) {
        SafeCleanTemp();
        if (mycmd.cmd_findPara("-ll") != -1) parlv.isMergeGhost = false;
        if (mycmd.cmd_findPara("-gh") != -1) parlv.isMergeGhost = true;
        glv_temp = parlv.MergingPar2(id_glv);
        pushList(glv_temp);
    }
    return 0;
}

int CtrlLouvain::CtrlPar_Louvain_thread() {
    if (parlv.num_par <= 0 || parlv.st_Partitioned == false) {
        printf("\033[1;31;40mERROR\033[0m: Partitioned graphNews are not set!\n");
        return -1;
    }
    printf("\033[1;37;40mINFO\033[0m: Total %d sub-graphNews will be mapped to %d devices\n", parlv.num_par,
           this->num_dev);
#ifndef _SINGLE_THREAD_MULTI_DEV_
    std::thread td[this->num_dev];
    // clock_t stime[this->num_dev], etime[this->num_dev];
    {
        for (int dev = 0; dev < this->num_dev; dev++) {
            parlv.timesPar.timeLv_dev[dev] = omp_get_wtime();
            bool hasGhost = true;
            td[dev] = std::thread(LouvainGLV_general_batch_thread, hasGhost, flowMode, dev, id_glv, this->num_dev,
                                  parlv.num_par, parlv.timesPar.timeLv, parlv.par_src, parlv.par_lved, xclbinPath,
                                  numThreads, minGraphSize, threshold, C_threshold, this->isParallel, this->numPhase);
        }

        for (int dev = 0; dev < this->num_dev; dev++) {
            td[dev].join();
            parlv.timesPar.timeLv_dev[dev] = omp_get_wtime() - parlv.timesPar.timeLv_dev[dev];
            // std::cout<< "INFO: Thread " << dev << " need time is "<< (etime[d] - stime[d])/CLOCKS_PER_SEC << "s." <<
            // std::endl;
        }
    }
#else
    {
        parlv.timeLv_dev[0] = LouvainGLV_general_batch(0, this->num_dev, parlv.num_par, parlv.timeLv, parlv.par_src,
                                                       parlv.par_lved, xclbinPath, numThreads, id_glv, minGraphSize,
                                                       threshold, C_threshold, this->isParallel, this->numPhase);

        parlv.timeLv_dev[1] = LouvainGLV_general_batch(1, this->num_dev, parlv.num_par, parlv.timeLv, parlv.par_src,
                                                       parlv.par_lved, xclbinPath, numThreads, id_glv, minGraphSize,
                                                       threshold, C_threshold, this->isParallel, this->numPhase);
    }
#endif
    for (int p = 0; p < parlv.num_par; p++) {
        pushList(parlv.par_lved[p]);
        id_glv++;
    }
    parlv.timesPar.timeLv_all = 0;
    for (int d = 0; d < parlv.num_dev; d++)
        if (parlv.timesPar.timeLv_all < parlv.timesPar.timeLv_dev[d])
            parlv.timesPar.timeLv_all = parlv.timesPar.timeLv_dev[d];

    parlv.st_ParLved = true;
    // parlv.PreMerge();
    // parlv.CheckGhost();
    return 0;
}

int CtrlLouvain::ParLV_thread(GLV* src, int num_par, bool isPrun, int th_prun) {
    if (parlv.num_par <= 0 || parlv.st_Partitioned == true) {
        printf("\033[1;31;40mERROR\033[0m: Partitioned graphNews are not set!\n");
        return -1;
    }
// printf("\033[1;37;40mINFO\033[0m: Total %d sub-graphNews will be mapped to %d devices\n",  parlv.num_par,
// this->num_dev);
#ifndef _SINGLE_THREAD_MULTI_DEV_
    std::thread td[this->num_dev];
    parlv.Init(flowMode, src, num_par, this->num_dev);
    for (int dev = 0; dev < this->num_dev; dev++) {
        parlv.timesPar.timeLv_dev[dev] = omp_get_wtime();
        td[dev] = std::thread(ParLV_general_batch_thread,
                              // parlv.timeLv_dev[dev] = LouvainGLV_general_batch(
                              flowMode, src, dev, this->num_dev, parlv.num_par, parlv.timesPar.timeLv, parlv.par_src,
                              parlv.par_lved, xclbinPath, numThreads, minGraphSize, threshold, C_threshold,
                              this->isParallel, this->numPhase);
    }

    for (int dev = 0; dev < this->num_dev; dev++) {
        td[dev].join();
        parlv.timesPar.timeLv_dev[dev] = omp_get_wtime() - parlv.timesPar.timeLv_dev[dev];
    }
#else
#endif
    /*
    parlv.timeLv_dev[0] = LouvainGLV_general_batch(0, this->num_dev, parlv.num_par, parlv.timeLv, parlv.par_src,
    parlv.par_lved,
                    xclbinPath,  numThreads, id_glv, minGraphSize, threshold, C_threshold, this->isParallel,
    this->numPhase);

    parlv.timeLv_dev[1] = LouvainGLV_general_batch(1, this->num_dev, parlv.num_par, parlv.timeLv, parlv.par_src,
    parlv.par_lved,
                    xclbinPath,  numThreads, id_glv, minGraphSize, threshold, C_threshold, this->isParallel,
    this->numPhase);
*/
    for (int p = 0; p < parlv.num_par; p++) pushList(parlv.par_src[p]);
    for (int p = 0; p < parlv.num_par; p++) pushList(parlv.par_lved[p]);
    parlv.timesPar.timeLv_all = 0;
    for (int d = 0; d < parlv.num_dev; d++)
        if (parlv.timesPar.timeLv_all < parlv.timesPar.timeLv_dev[d])
            parlv.timesPar.timeLv_all = parlv.timesPar.timeLv_dev[d];
    parlv.st_Partitioned = true;
    parlv.st_ParLved = true;
    // parlv.PreMerge();
    // parlv.CheckGhost();
    return 0;
}
int CtrlLouvain::CtrlPar_Louvain() {
    return CtrlPar_Louvain_thread();
}

int CtrlLouvain::CtrlPar_PreMerge() {
    if (parlv.num_par <= 0 || parlv.st_ParLved == false) {
        printf("\033[1;31;40mERROR\033[0m: Partitioned graphNews are not st_ParLved!\n");
        return -1;
    }
    parlv.PreMerge();
    return 0;
}
int CtrlLouvain::exe_LV_HISTORY() {
    mycmd.PrintHistory();
    return 0;
}
int CtrlLouvain::exe_LV_MAN() {
    PrintMan();
    printf("For more information, please use: <cmd> --help\n");
    return 0;
}
int CtrlLouvain::exe_LV_RENAME() {
    if (mycmd.cmd_findPara("--help") != -1) return 1; // man_LV_PG();
    int id_name;
    GLV* pglv = findByCmd(id_name);
    if (pglv == NULL) {
        printf("\033[1;31;40mERROR\033[0m: exe_LV_RENAME::Object is NULL or id can not be found\n");
        return -1;
    }
    pglv->SetName(mycmd.argv[id_name]);
    return 0;
}
int CtrlLouvain::exe_LV_DEV() {
    std::vector<cl::Device> devices = xcl::get_xil_devices();
    num_dev = devices.size();
    printf("\033[1;37;40mINFO\033[0m: Number of devices: %d\n", num_dev);
    for (int d = 0; d < num_dev; d++) {
        std::string devName = devices[d].getInfo<CL_DEVICE_NAME>();
        printf("\033[1;37;40mINFO\033[0m: Found Device=%s\n", devName.c_str());
        std::cout << "\t\tDevice Vendor               : " << devices[d].getInfo<CL_DEVICE_VENDOR>() << std::endl;
        std::cout << "\t\tDevice Version              : " << devices[d].getInfo<CL_DEVICE_VERSION>() << std::endl;
        std::cout << "\t\tDevice Max Compute Units    : " << devices[d].getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>()
                  << std::endl;
        std::cout << "\t\tDevice Global Memory        : " << devices[d].getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>()
                  << std::endl;
        std::cout << "\t\tDevice Max Clock Frequency  : " << devices[d].getInfo<CL_DEVICE_MAX_CLOCK_FREQUENCY>()
                  << std::endl;
        std::cout << "\t\tDevice Max Memory Allocation: " << devices[d].getInfo<CL_DEVICE_MAX_MEM_ALLOC_SIZE>()
                  << std::endl;
        std::cout << "\t\tDevice Local Memory         : " << devices[d].getInfo<CL_DEVICE_LOCAL_MEM_SIZE>()
                  << std::endl;
        std::cout << "\t\tDevice Available            : " << devices[d].getInfo<CL_DEVICE_AVAILABLE>() << std::endl;
    }
    useKernel = num_dev > 0;
}
int CtrlLouvain::ExeCmd() {
    int cmd = mycmd.cmd_last;

    if (cmd == LVLOADG) {
        if (-1 != exe_LV_LOADG()) ShowPara();
    } else if (cmd == LVLOADBAT) {
        if (-1 != exe_LV_LOADBAT()) { /*exe_LV_STATUS();exe_LV_PL()*/
            ;
        }
    } else if (cmd == LVDEV) {
        exe_LV_DEV();
    } else if (cmd == LVMODUG) {
        exe_LV_MODUG();
    } else if (cmd == LVQUIT) {
        ;
    } else if (cmd == LVSTATUS) {
        exe_LV_STATUS();
    } else if (cmd == LVSETM) {
        if (-1 != exe_LV_SETM()) ShowPara();
    } else if (cmd == LVSETTH) {
        if (-1 != exe_LV_SETTH()) {
            ShowPara();
        }
    } else if (cmd == LVSETTHC) {
        if (-1 != exe_LV_SETTHC()) {
            ShowPara();
        }
    } else if (cmd == LVPG) {
        exe_LV_PG();
    } else if (cmd == LVRSTC) {
        if (-1 != exe_LV_RSTC()) {
            exe_LV_STATUS();
        }
    } else if (cmd == LVRSTCL) {
        if (-1 != exe_LV_RSTCL()) {
            exe_LV_STATUS();
            exe_LV_PL();
        }
    } else if (cmd == LVINFOG) {
        exe_LV_INFOG();
    } else if (cmd == LVUNIQ) {
        exe_LV_UNIQ();
    } else if (cmd == LVTEST) {
        if (-1 != exe_LV_TEST()) {
            ShowPara();
            exe_LV_PL();
        }
    } else if (cmd == LVPL) {
        exe_LV_PL();
    } else if (cmd == LVPUSH) {
        if (-1 != exe_LV_PUSH()) {
            ShowPara();
            exe_LV_PL();
        } // ShowPara();
    } else if (cmd == LVPOP) {
        if (-1 != exe_LV_POP()) {
            exe_LV_PL();
            ShowPara();
        }
    } else if (cmd == LVDEL) {
        if (-1 != exe_LV_DEL()) {
            exe_LV_PL();
            ShowPara();
        }
    } else if (cmd == LVDEMO) {
        exe_LV_DEMO();
    } else if (cmd == LV) {
        if (-1 != exe_LV_LV()) {
            ShowPara();
        }
    } else if (cmd == LVPAR) {
        if (-1 != exe_LV_PAR()) {
            ShowPara();
            exe_LV_PL();
        }
    } else if (cmd == LVHISTORY) {
        exe_LV_HISTORY();
    } else if (cmd == LVMAN) {
        exe_LV_MAN();
    } else if (cmd == LVRENAME) {
        if (-1 != exe_LV_RENAME()) {
            ShowPara();
            exe_LV_PL();
        }
    } else if (cmd == LVMERGE) {
        if (-1 != exe_LV_MERGE()) {
            ShowPara();
        }
    } else if (cmd == LVCAT) {
        if (-1 != exe_LV_CAT()) {
            ShowPara();
        }
    } else if (cmd == LVSAVE) {
        if (-1 != exe_LV_SAVE()) {
            ShowPara();
        }
    } else {
        ;
    }
    if (cmd != LVLOADBAT) // To avoid double push the last command which was already pushed when doing batch
        mycmd.his_cmd.push_back(mycmd.line_last);
    mycmd.argc = 0;
    return cmd;
}
int CtrlLouvain::Run() {
    int cmd;
    PrintMan();
    ////////////////////////
    // Main command loop  //
    ////////////////////////
    do {
        if (!glv_curr && !glv_temp && cnt_list == 0) {
            printf(
                "\033[1;37;40mINFO\033[0m::glv_curr and glv_tmp is NULL; list is empty, please loadg <path> to load G "
                "from a file\n");
        }
        cmd = mycmd.cmd_GetCmd();
        ExeCmd();
    } while (cmd != LVQUIT);
    printf("\nCTRLLV: CMD: LVQUIT BYE! \n");
    return 0;
}
void CtrlLouvain::SetGfile(char* nm) {
    assert(nm != 0);
    strcpy(GfilePath, nm);
}
void CtrlLouvain::PrintList() {
    list<GLV*>::iterator iter;
    iter = glv_list.begin();
    while (iter != glv_list.end()) {
        (*iter)->printSimple();
        iter++;
    }
}

long G2Raw(graphNew* G, long size_grid, unsigned char*& img) {
    if (G == 0) return -1;
    long NV = G->numVertices;
    long* vtxPtr = G->edgeListPtrs;
    edge* vtxInd = G->edgeList;
    long NE = G->numEdges;
    long size_img = (NV + size_grid - 1) / size_grid;
    if (size_img > 8196) {
        printf("Too large image with width is %d/n", size_img);
        return -1;
    } else {
        printf("G2Raw:  size_img = (%d+%d-1)/%d = %d\n", NV, size_grid, size_grid, size_img);
    }
    img = (unsigned char*)malloc(size_img * size_img);
    if (img == 0) return -1;
    for (int i = 0; i < size_img * size_img; i++) {
        img[i] = 0;
    }

    for (int v = 0; v < NV; v++) {
        long adj1 = vtxPtr[v];
        long adj2 = vtxPtr[v + 1];
        int degree = adj2 - adj1;
        for (int d = 0; d < degree; d++) {
            long x = vtxInd[adj1 + d].head / size_grid;
            long y = vtxInd[adj1 + d].tail / size_grid;
            long index = y * size_img + x;
            if (img[index] < 255) img[index]++;
        }
    }
    return size_img;
}

void WriteAdjMatrix2Raw(unsigned char* img, long size_img, char* nm_img) {
    char* surfix = ".raw";
    char nm_file[4096];
    strcpy(nm_file, nm_img);
    strcat(nm_file, surfix);
    FILE* fp;
    printf("WriteAdjMatrix2Raw:image %s with size %d/n", nm_file, size_img);
    fp = fopen(nm_file, "wb");
    long cnt = fwrite(img, size_img * size_img, 1, fp);
    fclose(fp);
    printf("WriteAdjMatrix2Raw: Total %d pixel wrote/n", cnt);
}
