import numpy as np
from math import pi 

import ufl
from dolfinx import fem, io, mesh, plot
from ufl import ds, dx, grad, inner, dS,jump,div
from mpi4py import MPI
from petsc4py import PETSc
from math import sqrt
from dolfinx.mesh import create_unit_interval
from dolfinx.io import XDMFFile
import sys
import os
from typing import Optional, Callable

from math import log,sqrt

if os.name == "nt":
    _clear_line_command = ""
else:
    _clear_line_command = "\33[2K"

sys.setrecursionlimit(10**6)

class LinearSolver():
    def __init__(self, mat : PETSc.Mat,
                 pre = None,
                 tol : float = None,
                 maxiter : int = 100, 
                 atol : float = None,
                 callback : Optional[Callable[[int, float],None]] = None,
                 callback_sol : Optional[Callable[[PETSc.Vec],None]] = None,
                 printrates : bool = False):
        if atol is None and tol is None:
            tol = 1e-12
        self.mat = mat
        #assert (freedofs is None) != (pre is None)
        self.pre = pre
        self.tol = tol
        self.atol = atol
        self.maxiter = maxiter
        self.callback = callback
        self.callback_sol = callback_sol
        self.printrates = printrates
        self.residuals = []
        self.iterations = 0

    def Solve(self, rhs: PETSc.Vec, sol: Optional[PETSc.Vec] = None,
              initialize : bool = True) -> PETSc.Vec:
        self.iterations = 0
        self.residuals = []
        if sol is None:
            #sol = rhs.create(comm=MPI.COMM_WORLD)
            sol,_ = self.mat.createVecs()
            initialize = True
        if initialize:
            #print("ayyayayaya")
            #help(sol)
            #print("sol = ", sol)
            #sol.copy(rhs)
            sol.set(0)
            #print(" lalala ")
        self.sol = sol 
        self._SolveImpl(rhs=rhs,sol=sol)
        return sol,self.residuals

    def CheckResiduals(self,residual):
        self.iterations += 1
        self.residuals.append(residual)
        if len(self.residuals) == 1:
            if self.tol is None:
                self._final_residual = self.atol
            else:
                self._final_residual = residual * self.tol
                if self.atol is not None:
                    self._final_residual = max(self._final_residual,self.atol)
        else:
            if self.callback is not None:
                self.callback(self.iterations,residual)
            if self.callback_sol is not None:
                self.callback_sol(self.sol)

        if self.printrates:
            print("{}{} iteration {}, residual = {}    ".format(_clear_line_command, self.name, self.iterations, residual), end="\n" if isinstance(self.printrates, bool) else self.printrates)
            if self.iterations == self.maxiter and residual > self._final_residual:
                print("{}WARNING: {} did not converge tol TOL".format(_clear_line_command, self.name))
        is_converged = self.iterations >= self.maxiter or residual <= self._final_residual
        if is_converged and self.printrates == "\r":
            print("{}{} {}converged in {} iterations to residual {}".format(_clear_line_command, self.name, "NOT" if residual >= self._final_residual else "", self.iterations, residual))
        return is_converged


class GMResSolver(LinearSolver):
    name = "GMRes"

    def __init__(self, *args,
                 innerproduct : Optional[Callable[[PETSc.Vec,PETSc.Vec],float]] = None,
                 restart : Optional[int] = None,
                 **kwargs):
        super().__init__(*args, **kwargs)
        if innerproduct is not None:
            self.innerproduct = innerproduct
            self.norm = lambda x: sqrt(innerproduct(x,x).real)
        else: 
            self.innerproduct = lambda x,y: y.dot(x)
            self.norm = lambda x: x.norm()
            self.restart = restart 
        #print("constructor complete")

    def _SolveImpl(self, rhs: PETSc.Vec, sol: PETSc.Vec):
        #print("a")
        is_complex = False 
        A, pre, innerproduct, norm = self.mat, self.pre, self.innerproduct, self.norm 
        n = rhs.size 
        m = self.maxiter
        sn = np.zeros(m)
        cs = np.zeros(m)
        if self.callback_sol is not None:
            sol_start = sol.create()
            sol.copy(sol_start)
        r,tmp = self.mat.createVecs()
        #A(sol, tmp)
        A.mult(sol,tmp)
        tmp.axpy(-1, rhs)
        tmp.scale(-1)
        #print("tmp.max() = ", tmp.max())
        pre(tmp, r)
        #help(r)
        #tmp.copy(r)
        #print("r.max() = ", r.max())
        #print("tmp.max() = ", tmp.max())
        Q = [] 
        H = []
        q_1,_ = self.mat.createVecs()
        Q.append(q_1)
        r_norm = norm(r)
        #print("r_norm = ", r_norm)
        if self.CheckResiduals(abs(r_norm)):
            #print("returning sol")
            return sol
        r.copy(Q[0])
        #Q[0].copy(r)
        Q[0].scale(1/r_norm)
        beta = np.zeros(m+1)
        beta[0] = r_norm

        def arnoldi(A,Q,k): 
            q,_ = A.createVecs() 
            #help(A)
            #print("Q[k].max() =", Q[k].max())
            A.mult(Q[k],tmp)
            #print("tmp.max() =", tmp.max())
            #A(Q[k],tmp)
            
            pre(tmp,q)
            
            #print("q.max() =", q.max())
            h = np.zeros(m+1) 
            for i in range(k+1):
                h[i] = innerproduct(Q[i],q)
                #help(q)
                q.axpy(-1*h[i],Q[i])
            h[k+1] = norm(q)
            if abs(h[k+1]) < 1e-12:
                #print("oh oh")
                return h, None
            q.scale(1.0/h[k+1].real)
            return h, q

        def givens_rotation(v1,v2):
            #print("v2 = ", v2)
            if v2 == 0:
                return 1,0
            elif v1 == 0: 
                return 0, v2/abs(v2)
            else:
                t = sqrt( (v1.conjugate() * v1 + v2.conjugate() * v2).real )
                cs = abs(v1)/t 
                sn = v1/abs(v1) * v2.conjugate()/t
                return cs,sn
        
        def apply_givens_rotation(h, cs, sn, k):
            for i in range(k):
                temp = cs[i] * h[i] + sn[i] * h[i+1]
                h[i+1] = -sn[i].conjugate() * h[i] + cs[i].conjugate() * h[i+1]
                h[i] = temp
            cs[k], sn[k] = givens_rotation(h[k], h[k+1])
            h[k] = cs[k] * h[k] + sn[k] * h[k+1]
            h[k+1] = 0

        def calcSolution(k):
            # if callback_sol is set we need to recompute solution in every step 
            if self.callback_sol is not None: 
                sol.copy(sol_start) 
            mat = np.zeros((k+1,k+1))
            for i in range(k+1):
                mat[:,i] = H[i][:k+1]
            rs = np.zeros(k+1)
            rs[:] = beta[:k+1]
            y = np.linalg.solve(mat,rs)
            for i in range(k+1):
                sol.axpy(y[i],Q[i])

        #print("m = ", m)
        for k in range(m):
            #print("k = ", k)
            h,q  = arnoldi(A,Q,k)
            H.append(h)
            if q is None:
                break 
            Q.append(q)
            apply_givens_rotation(h,cs,sn,k)
            beta[k+1] = -sn[k].conjugate() * beta[k]
            beta[k] = cs[k] * beta[k] 
            error = abs(beta[k+1])
            if self.callback_sol is not None:
                calcSolution(k)
            if self.CheckResiduals(error):
                #print("breaking")
                break 
            if self.restart is not None and (k+1 == self.restart and not (self.restart == self.maxiter)):
                calcSolution(k) 
                del Q 
                restarted_solver = GMResSolver(mat = self.mat,
                                               pre = self.pre, 
                                               tol = 0,
                                               atol = self._final_residual, 
                                               callback = self.callback,
                                               callback_sol = self.callback_sol,
                                               maxiter = self.maxiter,
                                               restart = self.restart, 
                                               printrates = self.printrates)
                restarted_solver.iterations = self.iterations
                sol = restarted_solver.Solve(rhs = rhs, sol = sol, initialize = False)
                self.residuals += restarted_solver.residuals
                self.iterations = restarted_solver.iterations
                return sol
        calcSolution(k)
        return sol



def GMRes(A,b,pre=None,x=None,maxsteps = 100, tol = None, innerproduct = None, 
          callback = None, restart = None, startiteration = 0, printrates = True, reltol = None):
    solver = GMResSolver(mat=A,pre=pre,
                         maxiter=maxsteps, tol=reltol, atol=tol,
                         innerproduct=innerproduct,
                         callback_sol=callback, restart=restart,
                         printrates=printrates)
    return solver.Solve(rhs=b,sol=x)


class MinResSolver(LinearSolver):

    name = "MinRes"
    def __init__(self, *args,
                 innerproduct : Optional[Callable[[PETSc.Vec,PETSc.Vec],float]] = None,
                 **kwargs): 
        super().__init__(*args, **kwargs)
        if innerproduct is not None:
            self.innerproduct = innerproduct
            self.norm = lambda x: sqrt(innerproduct(x,x).real)
        else: 
            self.innerproduct = lambda x,y: y.dot(x)
            self.norm = lambda x: x.norm()

    def _SolveImpl(self, rhs: PETSc.Vec, sol: PETSc.Vec):
        pre, mat, u   = self.pre, self.mat, sol

        #innerproduct = lambda x,y: y.dot(x)
        innerproduct = self.innerproduct

        v_new,v = self.mat.createVecs()
        v_old,v_new2 = self.mat.createVecs()
        w,w_new = self.mat.createVecs()
        w_old,mz = self.mat.createVecs()
        z,z_new = self.mat.createVecs()
        tmp,_ = self.mat.createVecs()
         
        #def innerproduct(x,y):
        #    pre(y,tmp)
        #    return x.dot(tmp)
        
        mat.mult(u,v)
        v.axpy(-1, rhs)
        v.scale(-1)

        pre(v, z)

        # First step 
        gamma = sqrt(innerproduct(z,v))
        gamma_new = 0 
        z.scale(1/gamma)
        v.scale(1/gamma)
        
        ResNorm = gamma 
        ResNorm_old = gamma

        print("ResNorm = ", ResNorm)

        if self.CheckResiduals(ResNorm):
            return 

        eta_old = gamma 
        c_old = 1
        c = 1
        s_new = 0
        s = 0 
        s_old = 0 

        v_old.scale(0.0)
        w_old.scale(0.0)
        w.scale(0.0)

        k = 1 
        while True:
            mat.mult(z,mz)
            delta = innerproduct(mz,z)
            mz.copy(v_new)
            v_new.axpy(-delta, v)
            v_new.axpy(-gamma,v_old)

            pre(v_new,z_new)

            gamma_new = sqrt(innerproduct(z_new,v_new))
            z_new.scale(1/gamma_new)
            v_new.scale(1/gamma_new)

            alpha0 = c * delta - c_old * s * gamma
            alpha1 = sqrt(alpha0 * alpha0 + gamma_new * gamma_new)
            alpha2 = s * delta + c_old * c * gamma
            alpha3 = s_old * gamma 

            c_new = alpha0/alpha1
            s_new = gamma_new/alpha1

            z.copy(w_new)
            w_new.axpy(-alpha3,w_old)
            w_new.axpy(-alpha2,w)
            w_new.scale(1/alpha1)

            u.axpy(c_new * eta_old,w_new)
            eta = -s_new * eta_old 

            #update of residuum
            ResNorm = abs(s_new) * ResNorm_old 
            if self.CheckResiduals(ResNorm):
                return 
            k += 1

            # shift vectors by renaming 
            v_old, v, v_new = v, v_new, v_old 
            w_old, w, w_new = w, w_new, w_old
            z, z_new = z_new, z 

            eta_old = eta 

            s_old = s
            s = s_new 

            c_old = c 
            c = c_new 

            gamma = gamma_new 
            ResNorm_old = ResNorm 

def MinRes(mat, rhs, pre=None, sol=None, maxsteps = 100, printrates = True, initialize = True, tol = 1e-7,innerproduct = None):
    return MinResSolver(mat=mat, pre=pre, maxiter=maxsteps, 
                        printrates=printrates, 
                        tol=tol, innerproduct = innerproduct).Solve(rhs=rhs, sol=sol, 
                                       initialize=initialize)


























def pre(x,y):
    x.copy(y)


