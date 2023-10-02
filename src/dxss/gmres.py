import os
import sys
from abc import ABC, abstractmethod, abstractproperty
from math import sqrt
from typing import Callable, Optional

import numpy as np
from numpy.typing import NDArray
from petsc4py import PETSc

_clear_line_command = "" if os.name == "nt" else "\x1b[2K"

sys.setrecursionlimit(10**6)


class LinearSolver(ABC):
    """
    Abstract linear solver class for solving systems of linear equations.

    Attributes:
        mat: The matrix, A, representing the system of linear equations.
        pre: The pre-conditioning function applied to the input vector.
        tol: The tolerance for convergence criteria (default: None). If neither tol nor atol are set then the default is tol = 1e-5.
        maxiter: The maximum number of iterations for the solver (default: 100).
        atol: The absolute tolerance for convergence criteria (default: None). If neither tol nor atol are set then the default is tol = 1e-8.
        callback: A callback function called after each iteration (default: None).
        callback_sol: A callback function called with the solution vector (default: None).
        printrates: If True, print convergence rates during solving (default: False).

    Methods:
        solve(self, rhs, sol=None, initialize=True):
            Solves the linear system represented by the matrix and the
            right-hand side vector.

        check_residuals(self, residual):
            Checks the residuals of the solution vector and returns True if the
            convergence criteria are met.
    """

    def __init__(
        self,
        mat: PETSc.Mat,
        pre: Optional[Callable[[PETSc.Vec, PETSc.Vec], None]] = None,
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

        # List to store residuals after each iteration.
        self.residuals: list[float] = []
        # Number of iterations performed by the solver.
        self.iterations: int = 0

    @abstractproperty
    def name(self):
        """Name of the linear solver."""
        # TODO: we should be able to remove this later and use self.__name__

    @abstractmethod
    def _solve_impl(self, rhs: PETSc.Vec, sol: PETSc.Vec) -> PETSc.Vec:
        """Method-specific solving function called by `LinearSolver.solve`."""

    def solve(
        self,
        rhs: PETSc.Vec,
        sol: Optional[PETSc.Vec] = None,
        initialize: bool = True,
    ) -> PETSc.Vec:
        """Solve the linear system Ax = b.

        Args:
            rhs: The right-hand side vector b.
            sol: The solution vector x. If not provided, a new vector will be created (default: None).
            initialize: Whether to initialize the solution vector to zero (default: True).

        Returns:
            The solution vector x.
        """
        self.iterations = 0
        self.residuals = []
        if sol is None:
            sol, _ = self.mat.createVecs()
            initialize = True
        if initialize:
            sol.set(0)
        self.sol = sol
        self._solve_impl(rhs=rhs, sol=sol)
        return sol, self.residuals

    def check_residuals(self, residual: float) -> bool:
        """Checks the residuals of the solution vector.

        Args:
            residual: The residual of the solution vector.

        Returns:
            True if the convergence criteria are met.
        """
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

        if self.printrates and self._final_residual:
            print(
                f"{_clear_line_command}{self.name} iteration {self.iterations}, residual = {residual}    ",
                end="\n" if isinstance(self.printrates, bool) else self.printrates,
            )
            if self.iterations == self.maxiter and residual > self._final_residual:
                print(
                    f"{_clear_line_command}WARNING: {self.name} did not converge tol TOL",
                )
        is_converged = (
            self.iterations >= self.maxiter
            or self._final_residual is not None
            and residual <= self._final_residual
        )
        if (
            is_converged
            and self.printrates == "\r"
            and self._final_residual is not None
        ):
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
    """
    Implements the GMRES method for solving systems of linear equations.

    Attributes:
        name (str): The name of the solver ("GMRes").

    A concrete instance of the :class:`LinearSolver` class.
    """

    name = "GMRes"  # TODO: we shouldn't need this! Can get the self.__name__

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

        Called by the parent class' :meth:`solve` method. Implements the GMRes
        solver type, solving a linear system of equations Ax = b. Use the
        Arnoldi iteration to iteratively solve the system. Also handle special
        cases for callbacks and restarts.

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
        r, tmp = self.mat.createVecs()

        A = self.mat  # noqa: N806 | convention: Ax = b
        A.mult(sol, tmp)
        tmp.axpy(-1, rhs)
        tmp.scale(-1)
        if pre is not None:
            pre(tmp, r)
        Q = []  # noqa: N806
        q_1, _ = self.mat.createVecs()
        Q.append(q_1)
        r_norm = norm(r)
        if self.check_residuals(abs(r_norm)):
            return sol
        r.copy(Q[0])
        Q[0].scale(1 / r_norm)
        beta = np.zeros(self.maxiter + 1)
        beta[0] = r_norm

        def arnoldi(
            A: PETSc.Mat,  # noqa: N803
            Q: list[PETSc.Vec],  # noqa: N803
            k: int,
        ) -> tuple[NDArray, Optional[PETSc.Vec]]:
            """Perform the Arnoldi iteration for a given matrix A and a set of orthogonal vectors Q.

            Args:
                A: The matrix A.
                Q: The list of orthogonal vectors Q.
                k: The number of iterations.

            Returns:
                The computed values h and the computed vector q.
            """
            q, _ = A.createVecs()
            A.mult(Q[k], tmp)

            if pre is not None:
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

        def givens_rotation(v1: PETSc.Vec, v2: PETSc.Vec) -> tuple[float, float]:
            """Perform a Givens rotation on two given vectors.

            Args:
                v1: The first vector.
                v2: The second vector.

            Returns:
                A tuple representing the cosine and sine of the rotation angle.
            """
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

        def apply_givens_rotation(
            h: PETSc.Vec,
            cs: NDArray,
            sn: NDArray,
            k: int,
        ) -> None:
            """Apply Givens rotation to a given matrix.

            Args:
                h: The matrix to apply the Givens rotation to.
                cs: The array of cosine values for the rotation.
                sn: The array of sine values for the rotation.
                k: The index of the Givens rotation.
            """
            for i in range(k):
                temp = cs[i] * h[i] + sn[i] * h[i + 1]
                h[i + 1] = -sn[i].conjugate() * h[i] + cs[i].conjugate() * h[i + 1]
                h[i] = temp
            cs[k], sn[k] = givens_rotation(h[k], h[k + 1])
            h[k] = cs[k] * h[k] + sn[k] * h[k + 1]
            h[k + 1] = 0

        def calculate_solution(k: int) -> None:
            """
            Calculate the solution.

            Args:
                k: The iteration number.
            """
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
    A: PETSc.Mat,  # noqa: N803 | convention: Ax = b
    b: PETSc.Vec,
    pre=None,
    x: PETSc.Vec = None,
    maxsteps=100,
    tol=None,
    innerproduct=None,
    callback=None,
    restart=None,
    printrates=True,
    reltol=None,
) -> PETSc.Vec:
    """
    Solve a linear system using the GMRES method.

    Args:
        A: The coefficient matrix.
        b: The right-hand side vector.
        pre (optional): The preconditioner (default: None).
        x (optional): The initial guess (default: None).
        maxsteps (optional): The maximum number of iterations (default: 100).
        tol (optional): The absolute tolerance (default: None).
        innerproduct (optional): The inner product function (default: None).
        callback (optional): The callback function (default: None).
        restart (optional): The restart parameter (default: None).
        printrates (optional): Whether to print convergence rates (default: True).
        reltol (optional): The relative tolerance (default: None).

    Returns:
        The solution vector.
    """
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
    """A solver class for solving a system of linear equations using the Minimum Residual (MinRes) method.

    Attributes:
        name (str): The name of the solver, which is "MinRes".
    """

    name = "MinRes"  # TODO: remove this!

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

    def _solve_impl(self, rhs: PETSc.Vec, sol: PETSc.Vec) -> PETSc.Vec:  # noqa: PLR0915
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

        v_new, v = self.mat.createVecs()
        v_old, v_new2 = self.mat.createVecs()
        w, w_new = self.mat.createVecs()
        w_old, mz = self.mat.createVecs()
        z, z_new = self.mat.createVecs()
        tmp, _ = self.mat.createVecs()

        # def innerproduct(x,y):

        mat.mult(u, v)
        v.axpy(-1, rhs)
        v.scale(-1)

        if pre is not None:
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

            if pre is not None:
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
    """
    Solve a linear system using the Minimum Residual (MinRes) method.

    Args:
        mat: The coefficient matrix, A.
        rhs: The right-hand side vector, b, of the system of equations.
        pre: The preconditioner (optional, default: None).
        sol: The initial solution vector, x. (optional).
        maxsteps: The maximum number of iterations (optional).
        printrates: Whether to print convergence rates (optional, default: True).
        initialize: Whether to initialize the solver. (optional, default: True)
        tol: The absolute tolerance (optional).
        innerproduct: The inner product function (optional).

    Returns:
        The solution vector.
    """
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
