
#! /usr/bin/env python3
import sys
import os
import re

addr_name_list = ["layer_param_buf", "CONV_ONE_VEC", "wlut_ping"]
reg_map = {'commit_id':0x20,  'CPB': 0x24, 'BAT_NUM': 0x26, 'AIE_ACTIVATIONS_EN': 0x28, 'PL_FREQ': 0x2A, 'CHIP': 0x2C}
GRAPH_SYS_ADDR = 0x3800
GRAPH_STACK_SIZE = 0x400
GRAPH_SYNCBUF_SIZE = 0x20

AIE_ARRAY_OFFSET = 0x20000000000
COL_NUM = 50
ROW_NUM = 8

def getAddrMap(fname):
#  print '[INFO]: Parsing original file '+fname
  fa = open(fname, 'r')
  addr_map = {}
  for line in fa.readlines():
    for name in addr_name_list:
      if name in line:
        tline = line.strip().split()
        if len(tline) != 1:
          #print line
          addr_s = int(tline[0].split("..")[0], 16)
          addr_e = int(tline[0].split("..")[1], 16)
          ele_num = int(tline[2])
          addr_map[name] = [ele_num, addr_s, addr_e]
  return addr_map

def genMapCom(comMap, col, row):
  graph_base = GRAPH_SYS_ADDR + GRAPH_STACK_SIZE + GRAPH_SYNCBUF_SIZE
  aiecompiler_base = comMap['layer_param_buf'][1]
  hardware_base = AIE_ARRAY_OFFSET + int(col)*2**23 + int(row+1)*2**18 + graph_base
  #print '[INFO]: addr map in diff addr space'
  graph_map = {}
  aiecompiler_map = {}
  hardware_map = {}
  for name in comMap:
     diff = comMap[name][1] - aiecompiler_base
     graph_map[name] = graph_base + diff
     aiecompiler_map[name] = aiecompiler_base + diff
     hardware_map[name] = hardware_base + diff
  form = '  {:18}\t{:10}\t{:10}\t{:10}\n'
  pStr = form.format('name', 'graph_conv', 'aiecompiler', 'hardware')
  for name in graph_map:
     pStr += form.format(name, hex(graph_map[name]), hex(aiecompiler_map[name]), hex(hardware_map[name]))
  #if aiecompiler_map.has_key("CONV_ONE_VEC"):
  if "CONV_ONE_VEC" in aiecompiler_map:
   pStr += 'reg map\n'
   for name in reg_map:
     base = "CONV_ONE_VEC"
     off = reg_map[name]
     pStr += form.format(name, hex(graph_map[base] + off), hex(aiecompiler_map[base] + off), hex(hardware_map[base] + off))
  return pStr

def getAddr(start_col, start_row):
  col = start_col
  row = start_row
  mapfname = "./Work/aie/"+str(col) + "_"+str(row)+"/Release/"+str(col)+"_"+str(row)+".map"
  map_A = getAddrMap(mapfname)
  cfg = '----------------------------------------------------\n'
  cfg += 'addr_map of core (col, row)=(' + str(col) + ','+str(row)+')\n'
  '''
  form = '  {:18}\t{:8}\t{:10}\t{:10}\n'
  pStr = form.format('name', 'size(B)', 'start addr', 'end addr')
  for k in map_A:
    pStr += form.format(k, map_A[k][0],hex(map_A[k][1]),hex(map_A[k][2]))
  #print '[INFO]: addr map from original file '+mapfname
  #print pStr
  '''
  cfg += genMapCom(map_A, start_col, start_row)
  print(cfg)

def matrix2string(mat):
  retStr = '----------------------------------------------------\n'
  retStr += 'AIE cores used_map:\n'
  used = 0
  for i in range(len(mat)):
    for j in range(len(mat[0])):
        retStr += "O " if mat[ROW_NUM-1-i][j] == 0 else "X "
        used += 0 if mat[i][j] == 0 else 1
    retStr += "\n"
  retStr += "  total: " + str(used)
  return retStr;
def getGraphColStart(gname):
  content = open(gname, 'r').read()
  tiles=re.findall("tile\([0-9]*, [0-9]\d*\)", content)
  cores_used = [[0 for i in range(COL_NUM)] for j in range(ROW_NUM)]
  for i in range(len(tiles)):
    col_idx = int(re.findall("[0-9]+", tiles[i])[0])
    row_idx = int(re.findall("[0-9]+", tiles[i])[1])
    cores_used[row_idx][col_idx] = 1
  print(matrix2string(cores_used))
  if len(tiles) >= 1:
    return re.findall("[0-9]+", tiles[0])[0]
  else:
    return -1

def show_help():
  print("usage: ./getAddr.py 0 <path/to/graph.h>\n")
  print("       ./getAddr.py 1 <col> <row>\n")

def main(argv):
    if(len(argv) < 3):
        show_help()
        sys.exit()
    mode = int(argv[1])
    row = 0
    col = -1
    if mode == 0:
        col = int(getGraphColStart(argv[2])) + 1
    elif mode == 1:
        col = int(argv[2])
        row = int(argv[3])
    else:
        show_help()
        sys.exit()
    if col != -1:
        getAddr(col, row)

if __name__ == "__main__":
	sys.exit(main(sys.argv))

# /*******************************************************************************
# /*                                                                         
# * Copyright 2019 Xilinx Inc.                                               
# *                                                                          
# * Licensed under the Apache License, Version 2.0 (the "License");          
# * you may not use this file except in compliance with the License.         
# * You may obtain a copy of the License at                                  
# *                                                                          
# *    http://www.apache.org/licenses/LICENSE-2.0                            
# *                                                                          
# * Unless required by applicable law or agreed to in writing, software      
# * distributed under the License is distributed on an "AS IS" BASIS,        
# * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
# * See the License for the specific language governing permissions and      
# * limitations under the License.                                           
# */
# *******************************************************************************/