import os
import sys
from math import sqrt
from typing import Callable, Optional

import numpy as np
from petsc4py import PETSc

_clear_line_command = "" if os.name == "nt" else "\x1b[2K"

sys.setrecursionlimit(10**6)


class LinearSolver:
    def __init__(
        self,
        mat: PETSc.Mat,
        pre=None,
        tol: Optional[float] = None,
        maxiter: int = 100,
        atol: Optional[float] = None,
        callback: Optional[Callable[[int, float], None]] = None,
        callback_sol: Optional[Callable[[PETSc.Vec], None]] = None,
        printrates: bool = False,
    ):
        self.mat = mat
        if atol is None and tol is None:
            tol = 1e-12
        self.pre = pre
        self.tol = tol
        self.atol = atol
        self.maxiter = maxiter
        self.callback = callback
        self.callback_sol = callback_sol
        self.printrates = printrates

        self.residuals: list[float] = []
        self.iterations: int = 0

    def _solve_impl(self, rhs: PETSc.Vec, sol: PETSc.Vec) -> PETSc.Vec:
        """
        Method-specific solving function called by `LinearSolver.solve`.

        This is a no-op, and is intended for derived classes to override.
        """
        raise NotImplementedError

    def solve(
        self,
        rhs: PETSc.Vec,
        sol: Optional[PETSc.Vec] = None,
        initialize: bool = True,
    ) -> PETSc.Vec:
        self.iterations = 0
        self.residuals = []
        if sol is None:
            sol, _ = self.mat.create_vectors()
            initialize = True
        if initialize:
            sol.set(0)
        self.sol = sol
        self._solve_impl(rhs=rhs, sol=sol)
        return sol, self.residuals

    def check_residuals(self, residual):
        self.iterations += 1
        self.residuals.append(residual)
        if len(self.residuals) == 1:
            if self.tol is None:
                self._final_residual = self.atol
            else:
                self._final_residual = residual * self.tol
                if self.atol is not None:
                    self._final_residual = max(self._final_residual, self.atol)
        else:
            if self.callback is not None:
                self.callback(self.iterations, residual)
            if self.callback_sol is not None:
                self.callback_sol(self.sol)

        if self.printrates:
            print(
                f"{_clear_line_command}{self.name} iteration {self.iterations}, residual = {residual}    ",
                end="\n" if isinstance(self.printrates, bool) else self.printrates,
            )
            if self.iterations == self.maxiter and residual > self._final_residual:
                print(
                    f"{_clear_line_command}WARNING: {self.name} did not converge tol TOL",
                )
        is_converged = (
            self.iterations >= self.maxiter or residual <= self._final_residual
        )
        if is_converged and self.printrates == "\r":
            print(
                "{}{} {}converged in {} iterations to residual {}".format(
                    _clear_line_command,
                    self.name,
                    "NOT" if residual >= self._final_residual else "",
                    self.iterations,
                    residual,
                ),
            )
        return is_converged


class GMResSolver(LinearSolver):
    name = "GMRes"

    def __init__(
        self,
        *args,
        innerproduct: Optional[Callable[[PETSc.Vec, PETSc.Vec], float]] = None,
        restart: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if innerproduct is not None:
            self.innerproduct = innerproduct
            self.norm = lambda x: sqrt(innerproduct(x, x).real)
        else:
            self.innerproduct = lambda x, y: y.dot(x)
            self.norm = lambda x: x.norm()
            self.restart = restart

    def _solve_impl(self, rhs: PETSc.Vec, sol: PETSc.Vec):  # noqa: C901, PLR0915
        """The internal solving subfunction for the GMRes solver type.

        Called by the parent class' solve method. Implements the GMRes solver
        type, solving a linear system of equations Ax = b. Use the Arnoldi
        iteration to iteratively solve the system. It also handles special cases
        for callbacks and restarts.

        Args:
            rhs: The b vector in Ax = b.
            sol: The vector to contain solutions, x.

        Returns:
            A vector containing solutions, x.
        """
        # TODO: when refactoring this needs to be reduced in complexity and
        # split into smaller functions. We are suppressing flake8/ruff warnings
        # for this function:
        #  - C901 the complexity score of this function is 20!
        #  - PLR0915 there are too many conditional statements.
        pre, innerproduct, norm = self.pre, self.innerproduct, self.norm
        sn = np.zeros(self.maxiter)
        cs = np.zeros(self.maxiter)
        if self.callback_sol is not None:
            sol_start = sol.create()
            sol.copy(sol_start)
        r, tmp = self.mat.create_vectors()

        A = self.mat  # noqa: N806 | convention: Ax = b
        A.mult(sol, tmp)
        tmp.axpy(-1, rhs)
        tmp.scale(-1)
        pre(tmp, r)
        Q = []  # noqa: N806
        q_1, _ = self.mat.create_vectors()
        Q.append(q_1)
        r_norm = norm(r)
        if self.check_residuals(abs(r_norm)):
            return sol
        r.copy(Q[0])
        Q[0].scale(1 / r_norm)
        beta = np.zeros(self.maxiter + 1)
        beta[0] = r_norm

        def arnoldi(A, Q, k):  # noqa: N803
            q, _ = A.createVecs()
            A.mult(Q[k], tmp)

            pre(tmp, q)

            h = np.zeros(self.maxiter + 1)
            for i in range(k + 1):
                h[i] = innerproduct(Q[i], q)
                q.axpy(-1 * h[i], Q[i])
            h[k + 1] = norm(q)
            if abs(h[k + 1]) < 1e-12:
                return h, None
            q.scale(1.0 / h[k + 1].real)
            return h, q

        def givens_rotation(v1, v2):
            # TODO: can the Givens rotation def, and application functions be
            # moved out of this into a general utilities library?
            if v2 == 0:
                return 1, 0
            elif v1 == 0:
                return 0, v2 / abs(v2)
            else:
                t = sqrt((v1.conjugate() * v1 + v2.conjugate() * v2).real)
                cs = abs(v1) / t
                sn = v1 / abs(v1) * v2.conjugate() / t
                return cs, sn

        def apply_givens_rotation(h, cs, sn, k):
            for i in range(k):
                temp = cs[i] * h[i] + sn[i] * h[i + 1]
                h[i + 1] = -sn[i].conjugate() * h[i] + cs[i].conjugate() * h[i + 1]
                h[i] = temp
            cs[k], sn[k] = givens_rotation(h[k], h[k + 1])
            h[k] = cs[k] * h[k] + sn[k] * h[k + 1]
            h[k + 1] = 0

        def calculate_solution(k):
            # if callback_sol is set we need to recompute solution in every step
            if self.callback_sol is not None:
                sol.copy(sol_start)
            mat = np.zeros((k + 1, k + 1))
            for i in range(k + 1):
                mat[:, i] = H[i][: k + 1]
            rs = np.zeros(k + 1)
            rs[:] = beta[: k + 1]
            y = np.linalg.solve(mat, rs)
            for i in range(k + 1):
                sol.axpy(y[i], Q[i])

        H = []  # noqa: N806
        for k in range(self.maxiter):
            h, q = arnoldi(A, Q, k)
            H.append(h)
            if q is None:
                break
            Q.append(q)
            apply_givens_rotation(h, cs, sn, k)
            beta[k + 1] = -sn[k].conjugate() * beta[k]
            beta[k] = cs[k] * beta[k]
            error = abs(beta[k + 1])
            if self.callback_sol is not None:
                calculate_solution(k)
            if self.check_residuals(error):
                break
            if self.restart is not None and (
                k + 1 == self.restart and self.restart != self.maxiter
            ):
                calculate_solution(k)
                del Q
                restarted_solver = GMResSolver(
                    mat=self.mat,
                    pre=self.pre,
                    tol=0,
                    atol=self._final_residual,
                    callback=self.callback,
                    callback_sol=self.callback_sol,
                    maxiter=self.maxiter,
                    restart=self.restart,
                    printrates=self.printrates,
                )
                restarted_solver.iterations = self.iterations
                sol = restarted_solver.solve(rhs=rhs, sol=sol, initialize=False)
                self.residuals += restarted_solver.residuals
                self.iterations = restarted_solver.iterations
                return sol
        calculate_solution(k)
        return sol


def get_gmres_solution(  # noqa: PLR0913
    A,  # noqa: N803 | convention: Ax = b
    b,
    pre=None,
    x=None,
    maxsteps=100,
    tol=None,
    innerproduct=None,
    callback=None,
    restart=None,
    printrates=True,
    reltol=None,
):
    # TODO: this function has too many arguments
    # https://refactoring.guru/smells/long-parameter-list perhaps some kind of
    # configuration object could apply here. E.g. the abs and relative
    # tolerances could be a tuple. TODO: remove the noqa that suppresses PLR0913
    # for this function when fixed.
    #
    # Suppression of N803 should probably stay.
    solver = GMResSolver(
        mat=A,
        pre=pre,
        maxiter=maxsteps,
        tol=reltol,
        atol=tol,
        innerproduct=innerproduct,
        callback_sol=callback,
        restart=restart,
        printrates=printrates,
    )
    return solver.solve(rhs=b, sol=x)


class MinResSolver(LinearSolver):
    name = "MinRes"

    def __init__(
        self,
        *args,
        innerproduct: Optional[Callable[[PETSc.Vec, PETSc.Vec], float]] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if innerproduct is not None:
            self.innerproduct = innerproduct
            self.norm = lambda x: sqrt(innerproduct(x, x).real)
        else:
            self.innerproduct = lambda x, y: y.dot(x)
            self.norm = lambda x: x.norm()

    def _solve_impl(self, rhs: PETSc.Vec, sol: PETSc.Vec):  # noqa: PLR0915
        """The internal solving subfunction for the MinRes solver type.

        Called by the parent class' solve method.

        Args:
            rhs: The b vector in Ax = b.
            sol: The vector to contain solutions, x.

        Returns:
            A vector containing solutions, x.
        """
        # TODO: when refactoring this needs to be reduced in complexity and
        # split into smaller functions. Also to ask Janosch: why we do line 333?
        # We are suppressing a flake8/ruff warning for this function:
        #  - PLR0915 "there are too many conditional statements"
        pre, mat, u = self.pre, self.mat, sol

        innerproduct = self.innerproduct

        v_new, v = self.mat.create_vectors()
        v_old, v_new2 = self.mat.create_vectors()
        w, w_new = self.mat.create_vectors()
        w_old, mz = self.mat.create_vectors()
        z, z_new = self.mat.create_vectors()
        tmp, _ = self.mat.create_vectors()

        # def innerproduct(x,y):

        mat.mult(u, v)
        v.axpy(-1, rhs)
        v.scale(-1)

        pre(v, z)

        # First step
        gamma = sqrt(innerproduct(z, v))
        gamma_new = 0.0
        z.scale(1 / gamma)
        v.scale(1 / gamma)

        res_norm = gamma
        res_norm_old = gamma

        print("ResNorm = ", res_norm)

        if self.check_residuals(res_norm):
            return

        eta_old = gamma
        c_old = 1.0
        c = 1.0
        s_new = 0.0
        s = 0.0
        s_old = 0.0

        v_old.scale(0.0)
        w_old.scale(0.0)
        w.scale(0.0)

        k = 1
        while True:
            mat.mult(z, mz)
            delta = innerproduct(mz, z)
            mz.copy(v_new)
            v_new.axpy(-delta, v)
            v_new.axpy(-gamma, v_old)

            pre(v_new, z_new)

            gamma_new = sqrt(innerproduct(z_new, v_new))
            z_new.scale(1 / gamma_new)
            v_new.scale(1 / gamma_new)

            alpha0 = c * delta - c_old * s * gamma
            alpha1 = sqrt(alpha0 * alpha0 + gamma_new * gamma_new)
            alpha2 = s * delta + c_old * c * gamma
            alpha3 = s_old * gamma

            c_new = alpha0 / alpha1
            s_new = gamma_new / alpha1

            z.copy(w_new)
            w_new.axpy(-alpha3, w_old)
            w_new.axpy(-alpha2, w)
            w_new.scale(1 / alpha1)

            u.axpy(c_new * eta_old, w_new)
            eta = -s_new * eta_old

            # update of residuum
            res_norm = abs(s_new) * res_norm_old
            if self.check_residuals(res_norm):
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
            res_norm_old = res_norm


def get_minres_solution(
    mat,
    rhs,
    pre=None,
    sol=None,
    maxsteps=100,
    printrates=True,
    initialize=True,
    tol=1e-7,
    innerproduct=None,
):
    return MinResSolver(
        mat=mat,
        pre=pre,
        maxiter=maxsteps,
        printrates=printrates,
        tol=tol,
        innerproduct=innerproduct,
    ).solve(rhs=rhs, sol=sol, initialize=initialize)


def pre(x, y):
    x.copy(y)
