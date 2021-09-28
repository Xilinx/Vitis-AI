#!/usr/bin/env python3
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
import scipy.sparse.linalg
import sys
import argparse
import pdb


def compare(x, x_golden, rtol=1e-3, atol=1e-5, eps=1e-8):
    matches = np.isclose(x, x_golden, rtol=rtol, atol=atol)
    rel_err = np.abs(x / (x_golden + eps) - 1)
    abs_err = np.abs(x - x_golden)
    if matches.sum() == x.size:
        return True, 1, rel_err.max(), abs_err.max()
    else:
        ratio = matches.sum() / x.size * 100
        return False, ratio, rel_err.max(), abs_err.max()


def ichol(a):
    dim = a.shape[0]
    L = np.zeros(a.shape)
    for i in range(dim):
        for j in range(i):
            v = a[i, j]
            if v == 0:
                continue
            for k in range(j):
                v -= L[i, k] * L[j, k]
            L[i, j] = v / L[j, j]
        v = a[i, i]
        if v == 0:
            continue
        for k in range(i):
            v -= L[i, k] * L[i, k]
        if v < 0:
            print(v)
            pdb.set_trace()
            sys.exit(1)
        L[i, i] = np.sqrt(v)

    return L


class CG_Solver:

    def __init__(
        self,
        A: np.ndarray,
        b: np.ndarray,
        maxIter: int = 1000,
        tol=1e-5,
        verify=False,
            debug=False):
        self.A = A
        self.b = b
        self.maxIter = maxIter
        self.tol = tol
        self.verify = verify
        self.debug = debug

    def solve(self, M: np.ndarray = None):
        if M is None:
            M = np.identity(self.A.shape[0])
        if self.debug:
            k = np.linalg.cond(self.A)
            print("Condition number of matrix A is: ", k)
        return self._solver(M)

    def solve_ICH(self):
        L = np.linalg.cholesky(self.A) * (self.A != 0)
        M = L @ L.transpose()
        M = np.linalg.inv(M)
        if self.debug:
            k = np.linalg.cond(M @ self.A)
            print(
                "Condition number after applying incomplete Cholesky preconditioner is: ",
                k)
        return self._solver(M)

    def solve_SSOR(self):
        D = np.diag(np.diag(self.A))
        DL = np.linalg.inv(np.tril(self.A))
        M = DL.transpose() @ D @ DL
        if self.debug:
            k = np.linalg.cond(M @ self.A)
            print("Condition number after applying SSOR preconditioner is: ", k)
        return self._solver(M)

    def solve_Jacobi(self):
        M = np.linalg.inv(np.diag(np.diag(self.A)))
        if self.debug:
            k = np.linalg.cond(M @ self.A)
            print("Condition number applying Jacobi preconditioner is: ", k)
        return self._solver(M)

    def _solver(self, M: np.ndarray):
        x = np.zeros(self.b.size)
        r = self.b - self.A @ x
        z = M @ r
        rz = r.transpose() @ z
        bound = np.linalg.norm(self.b) * self.tol
        p = np.copy(z)
        res = np.linalg.norm(r)
        if self.debug:
            print("Beginning: Res = %e, RZ = %e." % (res * res, rz))
        for iters in range(0, self.maxIter):
            if res < bound:
                break
            Apk = self.A @ p
            alpha = rz / np.dot(p.transpose(), Apk)
            x = x + alpha * p
            r = r - alpha * Apk
            z = M @ r
            res = np.linalg.norm(r)
            beta = 1 / rz
            rz = r.transpose() @ z
            beta = beta * rz
            p = z + beta * p
            if self.debug:
                print("Iter %d: Alpha=%f, Beta=%f, Res=%e, RZ=%e." %
                      (iters, alpha, beta, res * res, rz))
        if self.verify:
            x_golden, info = scipy.sparse.linalg.cg(
                self.A, self.b, M=M, tol=self.tol, maxiter=self.maxIter)
            if np.linalg.norm(r) > bound:
                print(
                    "WARNING: CG solver reaches maximum iteration and the result fails to converge.")
            if info > 0:
                print(
                    "WARNING: Scipy CG solver reaches maximum iteration, does not converge.")
            matches, ratio, rerr, aerr = compare(x, x_golden)
            if matches:
                print("Verified against scipy CG solver.")
            else:
                print(
                    "WARNING: %.2f %% matches the results from scipy CG solver." %
                    ratio)
                print("Max relative error is %f, max abosolute error is %f." %
                      (rerr, aerr))
        return x, iters


def main(args):
    N = args.dimension
    A = dense_SPD(N)

    x = 2 * (np.random.random(A.shape[0]) - 0.5)
    b = A @ x

    solver = CG_Solver(A, b, verify=True, maxIter=3000, tol=1e-8)

    x_cg, n_iter = solver.solve()
    print("No.iterations: ", n_iter)
    matches, ratio, rerr, aerr = compare(x_cg, x)
    if matches:
        print("Result from CG solver matches the solution.")
    else:
        print(
            "WARNING: %.2f %% matches the solution." % ratio)
        print("Max relative error is %f, max abosolute error is %f." %
              (rerr, aerr))
    print("=" * 80)


if __name__ == "__main__":
    from genMat import dense_SPD
    parser = argparse.ArgumentParser(
        description='Conjugate Gradient Solver.')
    parser.add_argument(
        '--dimension',
        type=int,
        default=500,
        help='Dense matrix dimension')
    args = parser.parse_args()
    main(args)
