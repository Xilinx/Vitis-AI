# Copyright 2019 Xilinx, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

 # Copyright 2019 Xilinx, Inc.
 #
 # Licensed under the Apache License, Version 2.0 (the "License");
 # you may not use this file except in compliance with the License.
 # You may obtain a copy of the License at
 #
 #     http://www.apache.org/licenses/LICENSE-2.0
 #
 # Unless required by applicable law or agreed to in writing, software
 # distributed under the License is distributed on an "AS IS" BASIS,
 # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 # See the License for the specific language governing permissions and
 # limitations under the License.
 
import numpy as np
from collections import Counter
from matrix_params import * 

class signature:
    def __init__(self, parEntries, accLatency, channels, maxRows, maxCols, memBits):
        self.parEntries,self.accLatency,self.channels = parEntries,accLatency,channels
        self.maxRows,self.maxCols,self.memBits=maxRows,maxCols,memBits
        self.rbParam = row_block_param(memBits, channels)
        self.parParam = par_param(memBits, channels)
        self.nnzStore = nnz_store(memBits, parEntries, accLatency,channels)
        self.m,self.n,self.nnz = 0,0,0
        self.mPad,self.nPad,self.nnzPad = 0,0,0
    
    def add_spm(self, p_row, p_col, p_data, p_list):
        l_spm = sparse_matrix()
        l_spm.create_matrix(p_row, p_col, p_data)
        p_list.append(l_spm)
        return [l_spm.m,l_spm.n,l_spm.nnz,l_spm.minRowId,l_spm.minColId]

    def gen_rbs(self, p_spm):
        l_rbSpms = []
        self.m,self.n,self.nnz = p_spm.m,p_spm.n,p_spm.nnz
        self.mPad = p_spm.m
        self.nPad = math.ceil(p_spm.n / self.parEntries)*self.parEntries
        self.rbParam.add_dummyInfo()
        self.rbParam.totalRows=p_spm.m
        self.rbParam.totalRbs = 0
        l_numPars = 0
        l_sId,l_eId=0,0
        if p_spm.sort('r'):
            l_minRowId = p_spm.minRowId
            while l_eId < p_spm.nnz:
                l_row,l_col,l_data=[],[],[]
                if p_spm.row[p_spm.nnz-1] < l_minRowId + self.maxRows:
                    l_eId = p_spm.nnz
                else:
                    l_eId = np.flatnonzero(p_spm.row >= l_minRowId + self.maxRows)[0]
                if l_eId > l_sId:
                    l_row.extend(p_spm.row[l_sId:l_eId])
                    l_col.extend(p_spm.col[l_sId:l_eId])
                    l_data.extend(p_spm.data[l_sId:l_eId])
                    [l_m,l_n,l_nnz,l_spmMinRowId,l_spmMinColId] = self.add_spm(l_row,l_col,l_data,l_rbSpms)
                    assert l_m <= self.maxRows
                    l_sId = l_eId
                    self.rbParam.add_rbIdxInfo(l_spmMinRowId, l_spmMinColId, l_n, l_numPars)
                    self.rbParam.add_rbSizeInfo(l_m, l_nnz)
                    for i in range(6):
                        self.rbParam.add_dummyInfo()
                    self.rbParam.totalRbs += 1
                    if l_eId < p_spm.nnz:
                        l_minRowId = p_spm.row[l_eId]
                l_sId = l_eId
        else:
            print("ERROR: cannot sort matrix along rows")
        return l_rbSpms

    def gen_pars(self, p_rbSpms):
        l_parSpms = []
        l_totalPars = 0
        l_totalRbs = len(p_rbSpms)
        for i in range(l_totalRbs):
            l_rbSpm = p_rbSpms[i]
            assert l_rbSpm.nnz == self.rbParam.get_rbInfo(i,1)[1],"nnzs in rbParam != nnzs in rbSpm"
            assert l_rbSpm.m == self.rbParam.get_rbInfo(i,1)[0],"rows in rbParam != rows in rbSpm"
            assert l_rbSpm.m <= self.maxRows, "num of rows in rb > maxRows"
            l_rbPars = 0
            if l_rbSpm.sort_coo('c'):
                l_minColId = (l_rbSpm.minColId//self.parEntries) * self.parEntries
                l_sId,l_eId = 0,0
                while l_eId < l_rbSpm.nnz:
                    l_row,l_col,l_data=[],[],[]
                    if l_rbSpm.col[l_rbSpm.nnz-1] < l_minColId + self.maxCols:
                        l_eId = l_rbSpm.nnz
                    else:
                        l_eId = np.flatnonzero(l_rbSpm.col >= l_minColId + self.maxCols)[0]
                    if l_eId > l_sId:
                        l_row.extend(l_rbSpm.row[l_sId:l_eId])
                        l_col.extend(l_rbSpm.col[l_sId:l_eId])
                        l_data.extend(l_rbSpm.data[l_sId:l_eId])
                        [l_m,l_n,l_nnz,l_spmMinRowId,l_spmMinColId] = self.add_spm(l_row,l_col,l_data,l_parSpms)
                        assert l_m <= self.maxRows
                        assert l_n <= self.maxCols
                        l_sId = l_eId
                        l_rbPars += 1
                        if l_eId < l_rbSpm.nnz:
                            l_minColId = (l_rbSpm.col[l_eId]//self.parEntries)*self.parEntries
                self.rbParam.set_numPars(i, l_rbPars)
                l_totalPars += l_rbPars
            else:
                print("ERROR: cannot sort matrix along cols")
        return l_parSpms


    def pad_par(self, p_parSpm):
        l_nnzs = p_parSpm.nnz
        l_rows = p_parSpm.m
        l_parNnzs = 0
        p_parSpm.to_list()
        l_row,l_col,l_data = [],[],[]
        l_rowNnzs = Counter(p_parSpm.row)
        l_sId = 0
        while l_nnzs > 0:
            l_rowId = p_parSpm.row[l_sId]
            l_cRowNnzs = l_rowNnzs[l_rowId]
            l_nnzs -= l_cRowNnzs
            l_modId = 0
            l_idx = 0
            l_colIdBase = (p_parSpm.col[l_sId] // self.parEntries) * self.parEntries 
            l_rRowNnzs = 0 
            while l_idx < l_cRowNnzs:
                l_dataItem = p_parSpm.data[l_sId+l_idx]
                l_colId = p_parSpm.col[l_sId+l_idx]
                if l_modId == 0:
                    l_colIdBase = (l_colId // self.parEntries) * self.parEntries
                l_colIdMod = l_colId % self.parEntries
                if (l_colId != l_colIdBase+l_modId):
                    l_row.append(l_rowId)
                    l_col.append(l_colIdBase+l_modId)
                    l_data.append(0)
                else:
                    l_row.append(l_rowId)
                    l_col.append(l_colId)
                    l_data.append(l_dataItem)
                    l_idx += 1
                l_rRowNnzs += 1
                l_modId = (l_modId+1) % self.parEntries
            l_sId += l_cRowNnzs
           
            while (l_rRowNnzs % (self.parEntries * self.accLatency) !=0):
                l_row.append(l_rowId)
                l_col.append(l_colIdBase+l_modId)
                l_data.append(0)
                l_modId = (l_modId+1) % self.parEntries
                l_rRowNnzs += 1

        l_paddedParSpm = sparse_matrix()
        l_paddedParSpm.create_matrix(l_row, l_col, l_data)
        return l_paddedParSpm
 
    def gen_paddedPars(self, p_rbSpms):
        l_parSpms = self.gen_pars(p_rbSpms)
        l_paddedParSpms = []
        l_totalPars = len(l_parSpms)
        for i in range(l_totalPars):
            l_parSpm = l_parSpms[i]
            if l_parSpm.sort('r'):
                l_paddedSpm = self.pad_par(l_parSpm)
                assert l_paddedSpm.m <= self.maxRows
                assert l_paddedSpm.n <= self.maxCols,"num of cols > maxCols in partition"
                l_paddedParSpms.append(l_paddedSpm)
            else:
                print("ERROR: cannot sort matrix along rows")
        return l_paddedParSpms


    def gen_chPars(self, p_paddedParSpms):
        self.parParam.add_dummyInfo()
        l_chParSpms=[]
        for c in range(self.channels):
            l_chParSpms.append([]) 
        l_totalPars = len(p_paddedParSpms) 
        for i in range(l_totalPars):
            l_parSpm = p_paddedParSpms[i]
            assert l_parSpm.minColId % self.parEntries == 0
            l_baseParAddr = l_parSpm.minColId // self.parEntries
            l_colBks = math.ceil(l_parSpm.n / self.parEntries)
            l_rows = l_parSpm.m
            l_nnzs = l_parSpm.nnz
            l_nnzsPerCh = l_nnzs // self.channels
            self.nnzPad += l_nnzs
            l_sId,l_eId = 0,0
            l_chBaseAddr = np.zeros(self.channels, dtype=np.uint32)
            l_chCols = np.zeros(self.channels, dtype=np.uint32)
            l_chNnzs = np.zeros(self.channels, dtype=np.uint32)
            for c in range(self.channels):
                if (c == self.channels-1) or (l_sId+l_nnzsPerCh >= l_parSpm.nnz):
                    l_eId = l_parSpm.nnz
                else:
                    l_eId = l_sId + l_nnzsPerCh
                while (l_eId > 0) and (l_eId < l_parSpm.nnz) and (l_parSpm.row[l_eId] == l_parSpm.row[l_eId-1]):
                    l_eId += 1
                l_row = l_parSpm.row[l_sId:l_eId]
                l_col = l_parSpm.col[l_sId:l_eId]
                l_data= l_parSpm.data[l_sId:l_eId]
                l_sId = l_eId
                [l_m,l_n,l_nnz,l_minRowId,l_minColId] = self.add_spm(l_row, l_col, l_data, l_chParSpms[c])
                assert l_m <= self.maxRows
                assert l_n <= self.maxCols
                assert l_minColId % self.parEntries == 0
                assert l_n % self.parEntries == 0
                assert l_nnz % (self.parEntries * self.accLatency) == 0
                l_chBaseAddr[c] = (l_minColId // self.parEntries)-l_baseParAddr
                l_chCols[c] = l_n
                l_chNnzs[c] = l_nnz
            assert np.sum(l_chNnzs) == l_nnzs
            self.parParam.add_chInfo32(l_chCols)
            self.parParam.add_chInfo32(l_chNnzs)
            self.parParam.add_parInfo(l_baseParAddr, l_colBks, l_rows, l_nnzs)
            self.parParam.add_chInfo16(l_chBaseAddr)
            self.parParam.add_chInfo16(np.asarray(l_chCols) // self.parEntries)
            self.parParam.add_dummyInfo()
        self.parParam.totalPars = l_totalPars
        return l_chParSpms

    def update_rbParams(self, p_chParSpms):
        l_totalRbs = self.rbParam.totalRbs
        l_sRbParId = 0
        for rbId in range(l_totalRbs):
            [l_sRbRowId, l_minRbColId, l_rbCols, l_rbNumPars] = self.rbParam.get_rbInfo(rbId, 0)[0:4]
            l_chRbMinRowId = []
            l_chRbRows = [0] * self.channels
            l_chRbNnzs = [0] * self.channels
            for c in range(self.channels):
                l_minRowId = l_sRbRowId 
                l_endRowId = l_sRbRowId
                for parId in range(l_rbNumPars):
                    l_parId = l_sRbParId+parId
                    l_chParSpm = p_chParSpms[c][l_parId]
                    if (l_minRowId > l_chParSpm.minRowId) and (l_chParSpm.nnz != 0):
                        l_minRowId = l_chParSpm.minRowId
                    if l_endRowId < l_chParSpm.minRowId + l_chParSpm.m:
                        l_endRowId = l_chParSpm.minRowId + l_chParSpm.m
                    l_chRbNnzs[c] = l_chRbNnzs[c] + l_chParSpm.nnz
                l_chRbMinRowId.append(l_minRowId-l_sRbRowId)
                l_chRbRows[c] = l_endRowId - l_minRowId
                if (l_chRbRows[c] > self.maxRows):
                    print("rbId={}, channel={}, rbNumPars={}, l_sRbParId={}, l_parId={}\n".format(rbId,c,l_rbNumPars,l_sRbParId,l_parId))
                    print("l_endRowId = {}, l_minRowId = {}\n".format(l_endRowId, l_minRowId))
                assert l_chRbRows[c] <= self.maxRows
            self.rbParam.set_numNnzs(rbId, sum(l_chRbNnzs))
            self.rbParam.set_chInfo16(rbId, 0, l_chRbMinRowId)
            self.rbParam.set_chInfo16(rbId, 1, l_chRbRows)
            self.rbParam.set_chInfo32(rbId, l_chRbNnzs)                 
            l_sRbParId += l_rbNumPars

    def gen_nnzStore(self, p_chParSpms):
        for c in range(self.channels):
            self.nnzStore.add_dummyInfo(c)
        l_memIdxWidth = self.memBits//16
        l_rowIdxGap = self.parEntries * self.accLatency
        l_rowIdxMod = l_memIdxWidth * l_rowIdxGap
        l_colIdxMod = l_memIdxWidth * self.parEntries
        l_sParId = 0
        for rbId in range(self.rbParam.totalRbs):
            l_pars = self.rbParam.get_rbInfo(rbId, 0)[3]
            l_sRbRowId = self.rbParam.get_rbInfo(rbId, 0)[0]
            for parId in range(l_pars):
                l_parId = l_sParId + parId
                l_sParColId = self.parParam.get_parInfo(l_parId)[0]
                for c in range(self.channels):
                    l_chParSpm = p_chParSpms[c][l_parId]
                    self.nnzStore.totalRowIdxBks[c] += math.ceil(l_chParSpm.nnz/l_rowIdxMod)
                    self.nnzStore.totalColIdxBks[c] += math.ceil(l_chParSpm.nnz/l_colIdxMod)
                    self.nnzStore.totalNnzBks[c] += l_chParSpm.nnz // self.parEntries
                    l_sChRbRowId = self.rbParam.get_chInfo16(rbId, 0)[c] + l_sRbRowId
                    l_sChParColId = self.parParam.get_chInfo16(l_parId, 0)[c] + l_sParColId
                    l_rowIdx = [0]*l_memIdxWidth
                    l_colIdx = [0]*l_memIdxWidth
                    l_nnz=[0.0]*self.parEntries
                    for i in range(0,l_chParSpm.nnz,self.parEntries):
                        if i % l_rowIdxMod == 0:
                            for j in range(l_memIdxWidth):
                                if i+j*l_rowIdxGap < l_chParSpm.nnz:
                                    l_rowIdx[j] = l_chParSpm.row[i+j*l_rowIdxGap] - l_sChRbRowId
                            self.nnzStore.add_idxArr(c, l_rowIdx)
                        if i % l_colIdxMod == 0:
                            for j in range(l_memIdxWidth):
                                if i+j*self.parEntries < l_chParSpm.nnz:
                                    l_colIdx[j] = l_chParSpm.col[i+j*self.parEntries]//self.parEntries-l_sChParColId
                            self.nnzStore.add_idxArr(c, l_colIdx)
                        for j in range(self.parEntries):
                            l_nnz[j] = l_chParSpm.data[i+j]
                        self.nnzStore.add_nnzArr(c,l_nnz)
            l_sParId += l_pars

        for c in range(self.channels):
            self.nnzStore.totalBks[c] = self.nnzStore.totalRowIdxBks[c] + self.nnzStore.totalColIdxBks[c] + self.nnzStore.totalNnzBks[c]

    def process(self, mtxFullName, mtxName):
        l_spm = sparse_matrix()
        l_spm.read_matrix(mtxFullName, mtxName)
        l_rbSpms = self.gen_rbs(l_spm)
        assert self.rbParam.totalRows == l_spm.m,"total rows in rbParam != original total rows"
        del l_spm
        l_paddedParSpms = self.gen_paddedPars(l_rbSpms)
        del l_rbSpms
        l_chParSpms = self.gen_chPars(l_paddedParSpms)
        del l_paddedParSpms
        self.update_rbParams(l_chParSpms)
        self.gen_nnzStore(l_chParSpms)
        del l_chParSpms
        print("INFO: matrix {} parition done.".format(mtxName))
        print("      Original m, n, nnzs = {}, {}, {}".format(self.m, self.n, self.nnz))
        print("      After padding m, n, nnzs = {}, {}, {}".format(self.mPad, self.nPad, self.nnzPad))
        print("      Padding overhead is {}".format((self.nnzPad-self.nnz)/self.nnz))

    def store_rbParam(self, fileName):
        self.rbParam.write_file(fileName)

    def store_parParam(self, fileName):
        self.parParam.write_file(fileName)

    def store_nnz(self, fileNames):
        self.nnzStore.write_file(fileNames)

    def store_info(self, fileName):
        of = open(fileName, 'wb')
        int32Arr = [self.m,self.n,self.nnz,self.mPad,self.nPad,self.nnzPad]
        np.array(int32Arr).astype(np.uint32).tofile(of)
        of.close()
    
    def load_rbParam(self, fileName):
        self.rbParam.read_file(fileName)

    def load_parParam(self, fileName):
        self.parParam.read_file(fileName)

    def load_nnz(self, fileNames):
        self.nnzStore.read_file(fileNames)

    def load_info(self, fileName):
        fi = open(fileName, 'rb')
        int32Arr = np.fromfile(fi, dtype=np.uint32)
        self.m,self.n,self.nnz = int32Arr[0],int32Arr[1],int32Arr[2]
        self.mPad,self.nPad,self.nnzPad = int32Arr[3],int32Arr[4],int32Arr[5]
        fi.close()

    def print_rbParam(self, fileName):
        self.rbParam.print_file(fileName)

    def print_parParam(self, fileName):
        self.parParam.print_file(fileName)

    def print_nnz(self, fileNames):
        self.nnzStore.print_file(fileNames)

    def print_info(self, fileName):
        fo = open(fileName, 'w')
        fo.write("Original m, n, nnzs: {}, {}, {}\n".format(self.m, self.n, self.nnz))
        fo.write("Padded m, n, nnzs: {}, {}, {}\n".format(self.mPad, self.nPad, self.nnzPad))
        fo.close()

    def create_spm(self):
        l_memBytes = self.memBits//8
        l_totalPars = 0
        l_totalRbs = self.rbParam.totalRbs
        l_row,l_col,l_data=[],[],[]
        l_sParId = 0;
        l_chOffset=np.zeros(self.channels, dtype=np.uint32)
        l_chIdx = np.zeros(self.channels, dtype=np.uint32)
        l_chOffset += l_memBytes
        l_spmRow,l_spmCol,l_spmData = [],[],[]
        for rbId in range(l_totalRbs):
            [l_sRbRowId,l_minRbColId,l_rbCols,l_pars] = self.rbParam.get_rbInfo(rbId,0)[0:4]
            [l_rbRows, l_rbNnzs] = self.rbParam.get_rbInfo(rbId,1)[0:2]
            l_chRowOff = self.rbParam.get_chInfo16(rbId, 0)
            for c in range(self.channels):
                l_sChRowId = l_sRbRowId + l_chRowOff[c]
                for parId in range(l_pars):
                    l_parId = l_sParId + parId
                    [l_sParColId,l_parColBks,l_parRows,l_parNnzs]=self.parParam.get_parInfo(l_parId)[0:4]
                    l_chColOff = self.parParam.get_chInfo16(l_parId,0)[c]
                    l_sChColId = l_sParColId + l_chColOff
                    l_chNnzs = self.parParam.get_chInfo32(l_parId,1)[c]
                    [l_row,l_col,l_data,l_offset] = self.nnzStore.get_chPar(c, l_chOffset[c], 0, l_chNnzs, l_sChRowId, l_sChColId)
                    l_chOffset[c] = l_offset
                    l_chIdx[c] += l_chNnzs
                    for i in range(l_chNnzs):
                        if l_data[i] != 0:
                            l_spmRow.append(l_row[i])
                            l_spmCol.append(l_col[i])
                            l_spmData.append(l_data[i])
            l_sParId += l_pars
            l_totalPars += l_pars
        assert l_totalPars == self.parParam.totalPars
        l_spm = sparse_matrix()
        l_spm.create_matrix(l_spmRow,l_spmCol,l_spmData)
        return l_spm
    
    def check(self, mtxFullName, mtxName):
        l_spm = sparse_matrix()
        l_spm.read_matrix(mtxFullName, mtxName)
        l_sigSpm = self.create_spm()
        return l_spm.is_equal(l_sigSpm)
