from __future__ import annotations

import resource
import sys
import time
import warnings
from math import pi
from typing import Literal, NamedTuple

import numpy as np
import ufl
from dolfinx.mesh import create_unit_interval
from mpi4py import MPI
from petsc4py import PETSc

from dxss.gmres import get_gmres_solution
from dxss.space_time import (
    DataDomain,
    DataDomainIndicatorFunction,
    OrderSpace,
    OrderTime,
    SpaceTime,
    SpaceTimePETScMatrixWrapper,
    SpaceTimePETScPreconditionerWrapper,
    ValueAndDerivative,
    get_sparse_matrix,
)

try:
    import pypardiso
except ImportError:
    pypardiso = None

sys.setrecursionlimit(10**6)


def _get_lu_solver(msh, mat):
    solver = PETSc.KSP().create(msh.comm)
    solver.setOperators(mat)
    solver.setType(PETSc.KSP.Type.PREONLY)
    solver.getPC().setType(PETSc.PC.Type.LU)
    return solver


class PySolver:
    def __init__(self, Asp, psolver):  # noqa: N803 | convention Ax = b
        self.Asp = Asp
        self.solver = psolver
        if not pypardiso:
            warnings.warn(
                "Initialising a PySolver, but PyPardiso is not available.",
                stacklevel=2,
            )

    def solve(self, b_inp, x_out):
        self.solver._check_A(self.Asp)  # noqa: SLF001, TODO: fix this.
        b = self.solver._check_b(self.Asp, b_inp.array)  # noqa: SLF001, TODO: fix this.
        self.solver.set_phase(33)
        x_out.array[:] = self.solver._call_pardiso(  # noqa: SLF001, TODO: fix this.
            self.Asp,
            b,
        )


def omega_ind_convex(x):
    values = np.zeros(x.shape[1], dtype=PETSc.ScalarType)
    omega_coords = np.logical_or((x[0] <= 0.2), (x[0] >= 0.8))
    rest_coords = np.invert(omega_coords)
    values[omega_coords] = np.full(sum(omega_coords), 1.0)
    values[rest_coords] = np.full(sum(rest_coords), 0)
    return values


def omega_ind_nogcc(x):
    values = np.zeros(x.shape[1], dtype=PETSc.ScalarType)
    omega_coords = x[0] <= 0.2
    rest_coords = np.invert(omega_coords)
    values[omega_coords] = np.full(sum(omega_coords), 1.0)
    values[rest_coords] = np.full(sum(rest_coords), 0)
    return values


def sample_sol(t, xu):
    return ufl.cos(2 * pi * (t)) * ufl.sin(2 * pi * xu[0])


def dt_sample_sol(t, xu):
    return -2 * pi * ufl.sin(2 * pi * (t)) * ufl.sin(2 * pi * xu[0])


def _get_pypardiso_slab_solvers(space_time_solver):
    pardiso_slab_solver = pypardiso.PyPardisoSolver()
    sparse_slab_matrix = get_sparse_matrix(space_time_solver.get_slab_matrix())
    pardiso_slab_solver.factorize(sparse_slab_matrix)
    slab_solver = PySolver(sparse_slab_matrix, pardiso_slab_solver)
    pardiso_initial_slab_solver = pypardiso.PyPardisoSolver()
    sparse_initial_slab_matrix = get_sparse_matrix(
        space_time_solver.get_slab_matrix_first_slab(),
    )
    pardiso_initial_slab_solver.factorize(sparse_initial_slab_matrix)
    initial_slab_solver = PySolver(
        sparse_initial_slab_matrix,
        pardiso_initial_slab_solver,
    )
    return slab_solver, initial_slab_solver


def _get_petsc_lu_slab_solvers(space_time_solver):
    slab_solver = _get_lu_solver(
        space_time_solver.msh,
        space_time_solver.get_slab_matrix(),
    )
    initial_slab_solver = _get_lu_solver(
        space_time_solver.msh,
        space_time_solver.get_slab_matrix_first_slab(),
    )
    return slab_solver, initial_slab_solver


class SolverType(NamedTuple):
    method: Literal["gmres", "direct"] = "gmres"
    implementation: Literal["pypardiso", "petsc"] | None = None


def solve_problem(
    measure_errors: bool = False,
    plot_errors: bool = True,
    n_time_steps: int = 32,
    time_interval: float = 1.0,
    order: int = 1,
    n_cells: int | None = None,
    data_domain_indicator_function: DataDomainIndicatorFunction = omega_ind_convex,
    solver_type: SolverType | None = None,
) -> None:
    if solver_type is None:
        solver_type = SolverType("gmres", "petsc" if pypardiso is None else "pypardiso")
    n_cells = 5 * n_time_steps if n_cells is None else n_cells
    mesh = create_unit_interval(MPI.COMM_WORLD, n_cells)
    space_time_solver = SpaceTime(
        OrderTime(q=order, qstar=1 if order == 1 else 0),
        OrderSpace(k=order, kstar=1),
        N=n_time_steps,
        T=time_interval,
        t=0.0,
        msh=mesh,
        omega=DataDomain(indicator_function=data_domain_indicator_function),
        stabilisation_terms={
            "data": 1e4,
            "dual": 1.0,
            "primal": 1e-3,
            "primal-jump": 1.0,
        },
        solution=ValueAndDerivative(sample_sol, dt_sample_sol),
    )
    space_time_solver.setup_spacetime_finite_elements()
    space_time_solver.prepare_precondition_gmres()
    b_rhs = space_time_solver.get_spacetime_rhs()
    if solver_type.method == "gmres":
        slab_solver, first_slab_solver = (
            _get_pypardiso_slab_solvers(space_time_solver)
            if solver_type.implementation == "pypardiso"
            else _get_petsc_lu_slab_solvers(space_time_solver)
        )
        space_time_solver.set_solver_slab(slab_solver)
        space_time_solver.set_solver_first_slab(first_slab_solver)
        u_sol, _ = get_gmres_solution(
            A=SpaceTimePETScMatrixWrapper(space_time_solver),
            b=b_rhs,
            pre=SpaceTimePETScPreconditionerWrapper(space_time_solver),
            maxsteps=100000,
            tol=1e-7,
            printrates=True,
        )
    elif solver_type.method == "direct":
        A_space_time = space_time_solver.get_spacetime_matrix()  # noqa: N806
        u_sol, _ = A_space_time.createVecs()
        solver_space_time = _get_lu_solver(space_time_solver.msh, A_space_time)
        solver_space_time.solve(u_sol, b_rhs)
    else:
        msg = f"Unknown solver type method: {solver_type.method}."
        raise ValueError(msg)
    if plot_errors:
        space_time_solver.plot_error(u_sol, n_space=500, n_time_subdiv=20)
    if measure_errors:
        space_time_solver.measured_errors(u_sol)


if __name__ == "__main__":
    start_time = time.time()
    solve_problem(measure_errors=True, plot_errors=False)
    elapsed_time_seconds = time.time() - start_time
    memory_usage_bytes = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(f"Elapsed time  {elapsed_time_seconds} seconds")
    print(f"Memory usage in (GB) = {memory_usage_bytes / 1e6}")
