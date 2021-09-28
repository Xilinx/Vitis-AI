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

#ifndef _CTRLLV_H_
#define _CTRLLV_H_
#include "string.h"
#include "stdio.h"
#include "stdlib.h"
#include "xilinxlouvain.hpp"
#include "ParLV.h"
#include <list>
enum LVCMD {
    LVLOADG = 0,
    LVMODUG = 1,
    LVQUIT,
    LVSTATUS,
    LVSETM,
    LVSETTH = 5,
    LVSETTHC,
    LVPG,
    LVRSTCL,
    LVRSTC,
    LVINFOG,
    LVUNIQ,
    LVTEST,
    LVPL,
    LVPUSH,
    LVPOP,
    LVDEL,
    LV,
    LVPAR = 18,
    LVHISTORY,
    LVMAN,
    LVRENAME,
    LVDEMO,
    LVLOADBAT,
    LVMERGE,
    LVDEV,
    LVSAVE,
    LVCAT
};
#define NUM_CMD (28)
static const char* cmdlist[] = {"loadg", // 1
                                "modug", "quit", "st",
                                "setm", // 5
                                "setth", "setthc", "pg", "rstcl",
                                "rstc" // 10
                                ,
                                "infog" // 11
                                ,
                                "uniq" // 12
                                ,
                                "test" // 13 do anything testing here
                                ,
                                "pl" // 14 print TLV list
                                ,
                                "push" // 15
                                ,
                                "pop" // 16
                                ,
                                "del" // 17
                                ,
                                "lv" // 18 lv <curr> ; lv iter; lv renum; lv merge; lv csync ;
                                ,
                                "par" // 19
                                ,
                                "history" // 20
                                ,
                                "man" //
                                ,
                                "rename", "demo", "loadbat", "merge", "dev", "save", "cat"};

static const char* cmdlist_intro[] = {
    "loadg  : load a graphNew from the <path>.         1)cmd <path> ",
    "infog  : to be done. To print G's characters" // 11
    ,
    "st     : to show current status.               1)cmd", "setm   : to set minimal size for color -m.     1)cmd <v> ",
    "setth  : to set threshold of -t.               1)cmd <v>",
    "setthc : to set threshold of -d.               1)cmd <v>",
    "rstcl  : reset color by G.                     1)cmd 2)cmd tmp|temp|curr 3)cmd -id <id>",
    "rstc   : reset C by G.                         1)cmd 2)cmd tmp|temp|curr 3)cmd -id <id>" // 10
    ,
    "rename : reset name of GLV                     1)cmd 2)cmd tmp|temp|curr 3)cmd -id <id>",
    "modug  : compute modularity features.          1)cmd 2)cmd tmp|temp|curr 3)cmd -id <id>",
    "pg     : to print graphNew content.               1)cmd 2)cmd [-id <id>] [<start> <end>] 3)cmd [<start> <end>] "
    "[-id <id>] [-f [<path>]",
    "uniq   : to show information of part of graphNew. 1)cmd <start> <end>" // 12
    ,
    "par    : partition graphNew.                      1)cmd <start> <end> [-prun  [<number of ghost>]] [-num "
    "<num_par>] 2)par -lv|-merge" // 19,to adding '-m' for more method
    ,
    "pl     : print list status.                    1)cmd" // 14 print TLV list
    ,
    "push   : push a outside glv object into list.  1)cmd [tmp]2)cmd      cur|tmp|temp" // 15
    ,
    "pop    : pop out a glv object point from list. 1)cmd <id> 2)cmd <id> cur|tmp|temp" // 16
    ,
    "del    : delete a glv object.                  1)cmd <id>" // 17
    ,
    "lv     : run Louvain.                          1)cmd [-parallel] 2) cmd -history 3)cmd -par 4) -phase <n>",
    "history: print command history (and in file)   1)cmd 2) cmd -f <file>" // 20
    ,
    "man    : Manual of commands.                   1)cmd", "quit   : clean all glv(s) and quit.            1)cmd",
    "test   : testing.                              1)cmd" // 13 do anything testing here
    ,
    "demo   : To demonstrate basic flow             1)cmd",
    "loadbat: load a batch file and execute it      1)cmd <file>",
    "merge  : merge just one GLV by community       1)cmd <id>", "dev    : show device(s) info                   1)cmd",
    "saveplv: Saving glv into files: name_G|M|C.glv 1)cmd -f <file> 2)cmd -id <id> -f <file>",
    "cat    : cat two glv into glv_tmp              1)cmd -id <id1> -id<id1>"
    // todo
    ,
    "loadplv: Loading glv files: name_G|M|C.glv     1)cmd <path/name>",
    "clean  : To delete all glv objects             1)cmd"};

class TimeStt {
   public:
    double time;
    char* name;

   public:
    TimeStt(char* nm); //{name =nm; time = omp_get_wtime();}
    double End();      //{time = omp_get_wtime()-time; return time;};
    void Print();      //{printf("TIME_STT: %s =%lf sec\n",name, time);};
    void EndPrint();   //{End();Print();};
};

class myCmd {
   public:
    int argc;
    char argv[64][256];
    // char line_last[1024];
    int cmd_last;
    string line_last;
    int cnt_his;
    list<string> his_cmd;
    myCmd(); //{cnt_his=0; argc=0; cmd_last=-1;};
    char* cmd_SkipSpace(char* str);
    int cmd_CpyWord(char* des, char* src);
    int cmd_Getline();
    int cmd_Getline(char* str);
    int cmd_GetCmd();
    int cmd_GetCmd(char* str);
    void cmd_resetArg();
    int cmd_findPara(char* para);
    int PrintHistory();
};

class CtrlLouvain {
   public:
    // commonds: loadg, reload, status, setm, setcth, setth, pg, pcolor, pcom
    // input: argv, argc;
    CtrlLouvain();
    ~CtrlLouvain();
    void SetGfile(char*); //
    void SetPara();       //
    void ShowPara();
    void ClusterG();
    void ClusterOnce();
    int Run();
    void PrintMan();
    int ExeCmd();
    void CleanAll();
    void CleanList();
    int pushList(GLV*);
    GLV* CtrlParOne(GLV* src, long start, long end, bool isPrun, int th_prun);
    // void CtrlPar(GLV* src, int num_par, bool isPrun,  int th_prun);
    int ParLV_thread(GLV* src, int num_par, bool isPrun, int th_prun);
    double CtrlPar(GLV* src, int num_par, bool isPrun, int th_prun);
    int CtrlPar_Louvain();
    int CtrlPar_Louvain_thread();
    int CtrlPar_PreMerge();
    void print_glv_list();
    void print_parlv();

    int exe_LV_CAT();
    int exe_LV_DEL();
    int exe_LV_DEMO();
    int exe_LV_DEV();
    int exe_LV_HISTORY();
    int exe_LV_INFOG();
    int exe_LV_LV();
    int exe_LV_LOADG();
    int exe_LV_LOADBAT();
    int exe_LV_MODUG();
    int exe_LV_MAN();
    int exe_LV_PG();
    int man_LV_PG();
    int exe_LV_PL();
    int exe_LV_PUSH();
    int exe_LV_POP();
    int exe_LV_PAR();
    int exe_LV_QUIT();
    int exe_LV_RENAME();
    int exe_LV_RSTCL();
    int exe_LV_RSTC();
    int exe_LV_SAVE();
    int exe_LV_SETM();
    int exe_LV_SETTH();
    int exe_LV_SETTHC();
    int exe_LV_STATUS();
    int exe_LV_TEST();
    int exe_LV_UNIQ();
    int exe_LV_MERGE();

    int exe_batch(char* fileName);
    list<GLV*>::iterator IsInList(int id);
    GLV* IsInList_glv(int id);
    bool IsInList(GLV* it);
    bool IsTempInLIst();
    GLV* find(int id);
    GLV* findByCmd();
    GLV* findByCmd(int& i_p2);
    bool SafeCleanTemp();

    void ResetG();
    void PrintG(long star, long end);
    void PrintList();
    myCmd mycmd;
    char myArgc[64][256];
    char GfilePath[4096];
    char xclbinPath[4096];
    long minGraphSize;
    double threshold;   // default 0.0001
    double C_threshold; // default 0.000001
    int coloring;       // default =1;
    int numThreads;     // default =1;
    bool isParallel;    // default =0;
    int flowMode;       //
    int numPhase;

    GLV* glv_curr;
    GLV* glv_temp;
    list<GLV*> glv_list;
    int cnt_list; // How many glv points stored in list
    int id_glv;   // increasing forever for creating new ID for new GLV

    // LVRunBuff runBuff;
    ParLV parlv;
    // char   opts_xclbinPath[1024];
    bool useKernel;
    int num_dev;
};
long G2Raw(graphNew* G, long size_grid, unsigned char*& img);

void WriteAdjMatrix2Raw(unsigned char* img, long size_img, char* nm_file);
#endif
