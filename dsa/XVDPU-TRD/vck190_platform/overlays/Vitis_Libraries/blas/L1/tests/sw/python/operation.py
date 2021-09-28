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
import sys
import pdb

t_Debug = False


class OP_ERROR(Exception):
    def __init__(self, message):
        self.message = message


class OP:
    opDict = {
        'BLAS_L1': (
            'amax',
            'amin',
            'asum',
            'axpy',
            'swap',
            'scal',
            'dot',
            'copy',
            'nrm2'),
        'BLAS_L2': (
            'gemv',
            'gbmv',
            'sbmv',
            'symv',
            'spmv',
            'tpmv',
            'trmv',
            'tbmv')}

    @staticmethod
    def parse(opName):
        for cls, ops in OP.opDict.items():
            if opName in ops:
                return cls
        raise OP_ERROR("opName %s is not upperported." % opName)

    def __init__(self):
        pass

    def setDtype(self, dataType):
        self.dataType = dataType

    def setRtype(self, rtype):
        self.rtype = rtype


class DataGenerator:
    def __init__(self, engine=np.random.random):
        self.engine = engine

    def setRange(self, minValue, maxValue):
        self.minValue = minValue
        self.maxValue = maxValue

    def setDataType(self, dtype):
        self.dataType = dtype

    def data(self, size: tuple):
        v = self.engine(size) * (self.maxValue - self.minValue) + self.minValue
        v = v.astype(self.dataType)
        return v

    def scalar(self, value=None):
        tup = (1)
        v = self.data(tup)
        if value:
            v[0] = value
        return v

    def vector(self, size: int):
        tup = (size)
        return self.data(tup)

    def matrix(self, size: tuple):
        if not len(size) == 2:
            raise OP_ERROR("Matrix size error.")
        return self.data(size)

    def symmetricMatrix(self, size: int):
        tup = (size, size)
        mat = self.matrix(tup)
        mat = (mat + mat.transpose()) / 2
        return mat.astype(self.dataType)

    def symmetricBandedMatrix(self, size, k):
        mat = self.bandedMatrix(size, k)
        mat = (mat + mat.transpose()) / 2
        return mat.astype(self.dataType)

    def triangularMatrix(self, size: int, upper=True):
        mat = self.matrix((size, size))
        if upper:
            return np.triu(mat)
        else:
            return np.tril(mat)

    def bandedMatrix(self, size: tuple, k: tuple):
        if not len(size) == 2:
            raise OP_ERROR("Matrix size error.")
        if not len(k) == 2:
            raise OP_ERROR("Band dimention error.")
        m, n = size
        kl, ku = k
        matrix = self.matrix(size)
        for i in range(m):
            for j in range(n):
                if i - j > kl or i - j < -ku:
                    matrix[i][j] = 0
        return matrix

    def bandedMatrixStorage(
            self,
            matrix,
            k,
            upper=True,
            lower=True,
            colAlign=True):
        if upper == False and lower == False:
            raise OP_ERROR("Banded matrix storage setting error.")
        if not len(matrix.shape) == 2:
            raise OP_ERROR("Matrix dimention error.")
        m, n = matrix.shape
        if not len(k) == 2:
            raise OP_ERROR("Band dimention error.")
        kl, ku = k
        if colAlign:
            aDim = m + ku if m < n else n
            upperIndex = (ku if upper else 0)
            lowerIndex = (-kl if lower else 0)
            a = np.zeros(
                (upperIndex - lowerIndex + 1, aDim), dtype=self.dataType)
            for i in range(lowerIndex, upperIndex + 1):
                v = np.diag(matrix, i)
                startIndex = 0 if i < 0 else i
                a[upperIndex - i][startIndex:startIndex + len(v)] = v
        else:
            pass
        return a

    def symmetricMatrixStorage(self, matrix, upper=True, colMajor=False):
        if colMajor:
            pass
        else:
            if upper:
                return np.reshape(np.triu(matrix), -1)
            else:
                return np.reshape(np.tril(matrix), -1)

    def packedStorage(self, matrix, factor=1, upper=True):
        m, n = matrix.shape
        assert m == n
        a = list()
        for i in range(m):
            left = i if upper else 0
            right = m if upper else i + 1
            r = (right - left) % factor
            if not r == 0 and upper:
                a += [0] * (factor - r)
            for j in range(left, right):
                a.append(matrix[i][j])
            if not r == 0 and not upper:
                a += [0] * (factor - r)
        return np.asarray(a, dtype=matrix.dtype)


def dataGen(dataType, size, maxValue, minValue):
    a = np.random.random(size) * (maxValue - minValue) + minValue
    a = a.astype(dataType)
    return a


class BLAS_L1(OP):
    @staticmethod
    def parse(opName, maxV, minV):
        try:
            op = eval(opName)(BLAS_L1(opName, maxV, minV))
            return op
        except BaseException:
            raise OP_ERROR("%s is not defined." % opName)

    def __init__(self, name, maxV, minV):
        self.name = name
        self.maxV = maxV
        self.minV = minV
        self.interfaceList = ('p_x', 'p_y', 'p_xRes', 'p_yRes')

    def copyConstructor(self, object):
        self.name = object.name
        self.maxV = object.maxV
        self.minV = object.minV
        self.interfaceList = object.interfaceList

    def setSize(self, size):
        self.vectorDim = size
        self.sizeStr = "v%d" % size

    def compute(self):
        self.dataGen = DataGenerator()
        self.dataGen.setRange(self.minV, self.maxV)
        self.dataGen.setDataType(self.dataType)
        x = y = xr = yr = None
        alpha = self.dataGen.scalar()
        r = np.zeros(1, dtype=np.int32)
        return alpha, x, y, xr, yr, r

    def addInstr(self, blas_gen, register):
        for alpha, x, y, xr, yr, r in register:
            blas_gen.addB1Instr(
                self.name,
                self.vectorDim,
                alpha,
                x,
                y,
                xr,
                yr,
                r.astype(
                    self.rtype))

    def features(self):
        features = dict()
        features['Op name'] = self.name
        features['Vec. Size'] = self.vectorDim
        features['No.OPs'] = self.vectorDim
        return features

    def time(self, parallel, clock):
        return self.vectorDim * clock / parallel

    def test(self, runTest):
        vecDimList = runTest.profile['vectorDims']
        dataTypeList = runTest.dataTypes
        retTypeList = runTest.retTypes
        for dataType, retType in zip(dataTypeList, retTypeList):
            self.setDtype(dataType)
            self.setRtype(retType)
            runTest.build()
            for vecDim in vecDimList:
                self.setSize(vecDim)
                runTest.runTest()

    def paramTCL(self, f):
        f.write('   L1 true\n ')
        f.write('   opName "%s"\n ' % self.name)
        f.write('   vectorSize %d\n ' % self.vectorDim)


class amin(BLAS_L1):
    def __init__(self, blas_l1: BLAS_L1):
        self.copyConstructor(blas_l1)

    def compute(self):
        alpha, x, y, xr, yr, r = BLAS_L1.compute(self)
        x = self.dataGen.vector(self.vectorDim)
        r = np.argmin(np.abs(x))
        return alpha, x, y, xr, yr, r


class amax(BLAS_L1):
    def __init__(self, blas_l1: BLAS_L1):
        self.copyConstructor(blas_l1)

    def compute(self):
        alpha, x, y, xr, yr, r = BLAS_L1.compute(self)
        x = self.dataGen.vector(self.vectorDim)
        r = np.argmax(np.abs(x))
        return alpha, x, y, xr, yr, r


class asum(BLAS_L1):
    def __init__(self, blas_l1: BLAS_L1):
        self.copyConstructor(blas_l1)

    def compute(self):
        alpha, x, y, xr, yr, r = BLAS_L1.compute(self)
        x = self.dataGen.vector(self.vectorDim)
        r = np.sum(np.abs(x))
        return alpha, x, y, xr, yr, r


class axpy(BLAS_L1):
    def __init__(self, blas_l1: BLAS_L1):
        self.copyConstructor(blas_l1)

    def compute(self):
        alpha, x, y, xr, yr, r = BLAS_L1.compute(self)
        x = self.dataGen.vector(self.vectorDim)
        y = self.dataGen.vector(self.vectorDim)
        yr = alpha * x + y
        return alpha, x, y, xr, yr, r

    def features(self):
        features = BLAS_L1.features(self)
        features['No.OPs'] = self.vectorDim * 2
        return features


class copy(BLAS_L1):
    def __init__(self, blas_l1: BLAS_L1):
        self.copyConstructor(blas_l1)

    def compute(self):
        alpha, x, y, xr, yr, r = BLAS_L1.compute(self)
        yr = x = self.dataGen.vector(self.vectorDim)
        return alpha, x, y, xr, yr, r


class dot(BLAS_L1):
    def __init__(self, blas_l1: BLAS_L1):
        self.copyConstructor(blas_l1)

    def features(self):
        features = BLAS_L1.features(self)
        features['No.OPs'] = self.vectorDim * 2
        return features

    def compute(self):
        alpha, x, y, xr, yr, r = BLAS_L1.compute(self)
        x = self.dataGen.vector(self.vectorDim)
        y = self.dataGen.vector(self.vectorDim)
        r = np.dot(x, y)
        return alpha, x, y, xr, yr, r


class nrm2(BLAS_L1):
    def __init__(self, blas_l1: BLAS_L1):
        self.copyConstructor(blas_l1)

    def compute(self):
        alpha, x, y, xr, yr, r = BLAS_L1.compute(self)
        x = self.dataGen.vector(self.vectorDim)
        r = np.linalg.norm(x)
        return alpha, x, y, xr, yr, r

    def features(self):
        features = BLAS_L1.features(self)
        features['No.OPs'] = self.vectorDim * 2
        return features


class swap(BLAS_L1):
    def __init__(self, blas_l1: BLAS_L1):
        self.copyConstructor(blas_l1)

    def compute(self):
        alpha, x, y, xr, yr, r = BLAS_L1.compute(self)
        yr = x = self.dataGen.vector(self.vectorDim)
        xr = y = self.dataGen.vector(self.vectorDim)
        return alpha, x, y, xr, yr, r

    def features(self):
        features = BLAS_L1.features(self)
        features['No.OPs'] = self.vectorDim * 3
        return features


class scal(BLAS_L1):
    def __init__(self, blas_l1: BLAS_L1):
        self.copyConstructor(blas_l1)

    def compute(self):
        alpha, x, y, xr, yr, r = BLAS_L1.compute(self)
        x = self.dataGen.vector(self.vectorDim)
        xr = alpha * x
        return alpha, x, y, xr, yr, r


class BLAS_L2(OP):
    @staticmethod
    def parse(opName, maxV, minV):
        try:
            op = eval(opName)(BLAS_L2(opName, maxV, minV))
            return op
        except BaseException:
            raise OP_ERROR("%s is not defined." % opName)

    def __init__(self, name, maxV, minV):
        self.name = name
        self.maxV = maxV
        self.minV = minV
        self.ku = 0
        self.kl = 0
        self.m = 0
        self.n = 0
        self.interfaceList = ('p_a', 'p_x', 'p_y', 'p_aRes', 'p_yRes')

    def copyConstructor(self, object):
        self.name = object.name
        self.maxV = object.maxV
        self.minV = object.minV
        self.ku = object.ku
        self.kl = object.kl
        self.m = object.m
        self.n = object.n
        self.interfaceList = object.interfaceList

    def features(self):
        features = dict()
        features['Op name'] = self.name
        features['Mat. Size m'] = self.m
        features['Mat. Size n'] = self.n
        features['No.OPs'] = 2 * self.m * self.n + self.m * 3
        return features

    def time(self, parallel, clock):
        return self.m * self.n * clock / parallel

    def compute(self):
        self.dataGen = DataGenerator()
        self.dataGen.setRange(self.minV, self.maxV)
        self.dataGen.setDataType(self.dataType)
        a = x = y = ar = yr = None
        alpha = self.dataGen.scalar()
        beta = self.dataGen.scalar()
        return alpha, beta, a, x, y, ar, yr

    def addInstr(self, blas_gen, register):
        for alpha, beta, a, x, y, ar, yr in register:
            blas_gen.addB2Instr(self.name, self.m, self.n, self.kl, self.ku,
                                alpha, beta, a, x, y, ar, yr)

    def setSize(self, mn):
        self.matrixDim = tuple(mn)
        self.m = mn[0]
        self.n = mn[1]
        self.sizeStr = "m%d-n%d" % (self.m, self.n)
        self.memorySize = self.m * self.n

    def paramTCL(self, f):
        f.write('   L2 true\n ')
        f.write('   opName "%s"\n ' % self.name)
        f.write('   matrixSize %d\n ' % (self.m * self.n))
        f.write('   vectorSize %d\n ' % self.n)
        f.write('   memorySize %d\n ' % self.memorySize)

    def test(self, runTest):
        vecDimList = runTest.profile['matrixDims']
        dataTypeList = runTest.dataTypes
        self.PE = runTest.parEntries
        for dataType in dataTypeList:
            self.setDtype(dataType)
            runTest.build()
            for vecDim in vecDimList:
                self.setSize(vecDim)
                self.specTest(runTest)

    def specTest(self, runTest):
        runTest.runTest()


class gemv(BLAS_L2):
    def __init__(self, blas_l2: BLAS_L2):
        self.copyConstructor(blas_l2)

    def compute(self):
        alpha, beta, a, x, y, ar, yr = BLAS_L2.compute(self)
        matrix = self.dataGen.matrix(self.matrixDim)
        x = self.dataGen.vector(self.n)
        y = self.dataGen.vector(self.m)
        yr = alpha * np.matmul(matrix, x,
                               dtype=self.dataGen.dataType) + beta * y
        return alpha, beta, matrix, x, y, ar, yr


class gbmv(BLAS_L2):
    def __init__(self, blas_l2: BLAS_L2):
        self.copyConstructor(blas_l2)

    def setK(self, kul):
        self.k = tuple(kul)
        self.kl = kul[0]
        self.ku = kul[1]
        self.sizeStr = "m%d-n%d-l%d-u%d" % (self.m, self.n, self.kl, self.ku)
        self.memorySize = (self.kl + self.ku + 1) * self.n
        if self.ku < self.n - self.m:
            print("WARNING: Current matrix configuration has multiple zero columns")
        if self.m - self.n > self.kl:
            print("WARNING: Current matrix configuration has multiple zero rows")

    def features(self):
        features = BLAS_L2.features(self)
        features['kl'] = self.kl
        features['ku'] = self.ku
        features['No.OPs'] = 2 * self.m * (self.ku + self.kl + 1) + self.m * 3
        return features

    def time(self, parallel, clock):
        return self.m * (self.ku + self.kl + 1) * clock / parallel

    def compute(self):
        alpha, beta, a, x, y, ar, yr = BLAS_L2.compute(self)
        matrix = self.dataGen.bandedMatrix(self.matrixDim, self.k)
        a = self.dataGen.bandedMatrixStorage(matrix, self.k)
        x = self.dataGen.vector(self.n)
        y = self.dataGen.vector(self.m)
        yr = alpha * np.matmul(matrix, x,
                               dtype=self.dataGen.dataType) + beta * y
        t_Debug and print("Matrix:\n", matrix)
        t_Debug and print("Storage:\n", a)
        return alpha, beta, a, x, y, ar, yr

    def specTest(self, runTest):
        kulList = runTest.profile['kulList']
        for kul in kulList:
            self.setK(kul)
            runTest.runTest()


class symv(BLAS_L2):
    def __init__(self, blas_l2: BLAS_L2):
        self.copyConstructor(blas_l2)
        self.upper = True

    def setSize(self, m):
        self.matrixDim = (m, m)
        self.m = m
        self.n = m
        self.memorySize = self.m * self.n
        self.sizeStr = "m%d" % (self.m)

    def setStorage(self, upper=True):
        self.upper = upper
        if self.upper:
            self.sizeStr = self.sizeStr + "_upper"
            self.ku = self.m
        else:
            self.sizeStr = self.sizeStr + "_lower"
            self.kl = self.m

    def compute(self):
        alpha, beta, a, x, y, ar, yr = BLAS_L2.compute(self)
        matrix = self.dataGen.symmetricMatrix(self.m)
        a = self.dataGen.symmetricMatrixStorage(matrix, self.upper)
        x = self.dataGen.vector(self.n)
        y = self.dataGen.vector(self.m)
        yr = alpha * np.matmul(matrix, x,
                               dtype=self.dataGen.dataType) + beta * y
        return alpha, beta, a, x, y, ar, yr

    def features(self):
        features = dict()
        features['Op name'] = self.name
        features['Mat. Size m'] = self.m
        features['No.OPs'] = 2 * self.m * self.m + self.m * 3
        return features

    def time(self, parallel, clock):
        return self.m * self.m * clock / parallel

    def specTest(self, runTest):
        self.setStorage(runTest.profile['storage'])
        runTest.runTest()


class trmv(symv):
    def __init__(self, blas_l2: BLAS_L2):
        self.copyConstructor(blas_l2)
        self.upper = True

    def features(self):
        features = dict()
        features['Op name'] = self.name
        features['Mat. Size m'] = self.m
        features['No.OPs'] = self.m * self.m + self.m * 3
        return features

    def time(self, parallel, clock):
        return self.m * (self.m + 1) // 2 * clock / parallel

    def compute(self):
        alpha, beta, a, x, y, ar, yr = BLAS_L2.compute(self)
        matrix = self.dataGen.triangularMatrix(self.m, self.upper)
        x = self.dataGen.vector(self.n)
        y = self.dataGen.vector(self.m)
        yr = alpha * np.matmul(matrix, x,
                               dtype=self.dataGen.dataType) + beta * y
        return alpha, beta, matrix, x, y, ar, yr

    def features(self):
        features = symv.features(self)
        features['No.OPs'] = self.m * (self.m + 1) + self.m * 3
        return features

    def time(self, parallel, clock):
        return self.m * (1 + self.m) / 2 * clock / parallel


class tpmv(trmv):
    def __init__(self, blas_l2: BLAS_L2):
        self.copyConstructor(blas_l2)
        self.upper = True

    def setSize(self, m):
        self.matrixDim = (m, m)
        self.m = m
        self.n = m
        self.memorySize = (m + 1) * m // 2 + (self.PE - 1) * m // 2
        self.sizeStr = "m%d" % (self.m)

    def compute(self):
        alpha, beta, a, x, y, ar, yr = BLAS_L2.compute(self)
        matrix = self.dataGen.triangularMatrix(self.m, self.upper)
        a = self.dataGen.packedStorage(matrix, self.PE, self.upper)
        x = self.dataGen.vector(self.n)
        y = self.dataGen.vector(self.m)
        yr = alpha * np.matmul(matrix, x,
                               dtype=self.dataGen.dataType) + beta * y
        return alpha, beta, a, x, y, ar, yr


class spmv(symv):
    def __init__(self, blas_l2: BLAS_L2):
        self.copyConstructor(blas_l2)
        self.upper = True

    def setSize(self, m):
        self.matrixDim = (m, m)
        self.m = m
        self.n = m
        self.memorySize = (m + 1) * m // 2 + (self.PE - 1) * m // 2
        self.sizeStr = "m%d" % (self.m)

    def compute(self):
        alpha, beta, a, x, y, ar, yr = BLAS_L2.compute(self)
        matrix = self.dataGen.symmetricMatrix(self.m)
        a = self.dataGen.packedStorage(matrix, self.PE, self.upper)
        x = self.dataGen.vector(self.n)
        y = self.dataGen.vector(self.m)
        yr = alpha * np.matmul(matrix, x,
                               dtype=self.dataGen.dataType) + beta * y
        return alpha, beta, a, x, y, ar, yr


class sbmv(BLAS_L2):
    def __init__(self, blas_l2: BLAS_L2):
        self.copyConstructor(blas_l2)
        self.upper = True
        self.lower = True

    def setK(self, k: int):
        self.kl = k
        self.ku = k
        self.k = (k, k)
        self.memorySize = (k + 1) * self.n
        self.sizeStr = "m%d_k%d" % (self.m, k)

    def features(self):
        features = symv.features(self)
        k = self.ku if self.ku != 0 else self.kl
        features['k'] = k
        features['No.OPs'] = 2 * self.m * (k * 2 + 1) + self.m * 3
        return features

    def time(self, parallel, clock):
        k = self.ku if self.ku != 0 else self.kl
        return self.m * (k * 2 + 1) / parallel * clock

    def setSize(self, m: int):
        self.matrixDim = (m, m)
        self.m = m
        self.n = m

    def setStorage(self, s=(True, True)):
        self.upper = s[0]
        self.lower = s[1]
        if self.upper and self.lower:
            self.sizeStr = self.sizeStr + "_full"
        elif not self.upper:
            self.ku = 0
            self.sizeStr = self.sizeStr + "_lower"
        elif not self.lower:
            self.kl = 0
            self.sizeStr = self.sizeStr + "_upper"
        else:
            raise OP_ERROR("Storage setting error.")

    def compute(self):
        alpha, beta, a, x, y, ar, yr = BLAS_L2.compute(self)
        matrix = self.dataGen.symmetricBandedMatrix(self.matrixDim, self.k)
        a = self.dataGen.bandedMatrixStorage(
            matrix, self.k, upper=self.upper, lower=self.lower)
        x = self.dataGen.vector(self.n)
        y = self.dataGen.vector(self.m)
        yr = alpha * np.matmul(matrix, x,
                               dtype=self.dataGen.dataType) + beta * y
        return alpha, beta, a, x, y, ar, yr

    def specTest(self, runTest):
        kulList = runTest.profile['kList']
        for kul in kulList:
            self.setK(kul)
            self.setStorage(runTest.profile['storage'])
            runTest.runTest()


class tbmv(sbmv):
    def __init__(self, blas_l2: BLAS_L2):
        self.copyConstructor(blas_l2)

    def setStorage(self, upper=True):
        self.upper = upper
        if self.upper:
            self.sizeStr = self.sizeStr + "_upper"
            self.k = (0, self.ku)
            self.kl = 0
        else:
            self.sizeStr = self.sizeStr + "_lower"
            self.k = (self.kl, 0)
            self.ku = 0

    def features(self):
        features = sbmv.features(self)
        features['No.OPs'] = 2 * self.m * (self.kl + self.ku + 1) + self.m * 3
        return features

    def time(self, parallel, clock):
        return self.m * (self.kl + self.ku + 1) / parallel * clock

    def compute(self):
        alpha, beta, a, x, y, ar, yr = BLAS_L2.compute(self)
        matrix = self.dataGen.bandedMatrix(self.matrixDim, self.k)
        a = self.dataGen.bandedMatrixStorage(matrix, self.k)
        x = self.dataGen.vector(self.n)
        y = self.dataGen.vector(self.m)
        yr = alpha * np.matmul(matrix, x,
                               dtype=self.dataGen.dataType) + beta * y
        return alpha, beta, a, x, y, ar, yr


def main():
    dg = DataGenerator()
    dg.setRange(-16, 16)
    dg.setDataType(np.int8)
    matrix = dg.triangularMatrix(8, 0)
    a = dg.packedStorage(matrix, 4, 0)
    print(matrix)
    print(a)
#  opName = 'sbmv'
#  className = None
#  for cls, ops in OP.opDict.items():
#    if opName in ops:
#      className = cls
#      break
#
#  if className:
#    op  = eval(className).parse(opName, -64,   64)
#    op.setDtype(np.int32)
#    op.setSize(int(sys.argv[1]))
#    op.setK(int(sys.argv[2]))
#    op.setStorage(False)
#    alpha, beta, a, x, y, ar, yr = op.compute()


if __name__ == '__main__':
    main()
