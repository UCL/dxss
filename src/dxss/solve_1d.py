from __future__ import annotations

import argparse
import resource
import sys
import time
import warnings
from math import pi, sqrt
from typing import Literal, NamedTuple

import numpy as np
import ufl
from dolfinx.mesh import (
    CellType,
    Mesh,
    create_unit_cube,
    create_unit_interval,
    create_unit_square,
)
from mpi4py import MPI
from petsc4py import PETSc

from dxss.gmres import get_gmres_solution
from dxss.meshes import get_mesh_data_all_around
from dxss.space_time import (
    DataDomain,
    DataDomainIndicator,
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


def get_mesh_1d(n_cells: int) -> Mesh:
    return create_unit_interval(comm=MPI.COMM_WORLD, nx=n_cells)


def get_mesh_2d(n_cells: int) -> Mesh:
    return create_unit_square(comm=MPI.COMM_WORLD, nx=n_cells, ny=n_cells)


def get_mesh_3d(n_cells: int) -> Mesh:
    return create_unit_cube(
        comm=MPI.COMM_WORLD,
        nx=n_cells,
        ny=n_cells,
        nz=n_cells,
        cell_type=CellType.hexahedron,
    )


def data_domain_indicator_function_1d_convex(x):
    values = np.zeros(x.shape[1], dtype=PETSc.ScalarType)
    omega_coords = np.logical_or((x[0] <= 0.2), (x[0] >= 0.8))
    rest_coords = np.invert(omega_coords)
    values[omega_coords] = np.full(sum(omega_coords), 1.0)
    values[rest_coords] = np.full(sum(rest_coords), 0)
    return values


def data_domain_indicator_function_1d_nogcc(x):
    values = np.zeros(x.shape[1], dtype=PETSc.ScalarType)
    omega_coords = x[0] <= 0.2
    rest_coords = np.invert(omega_coords)
    values[omega_coords] = np.full(sum(omega_coords), 1.0)
    values[rest_coords] = np.full(sum(rest_coords), 0)
    return values


def data_domain_indicator_function_2d_convex(x):
    values = np.zeros(x.shape[1], dtype=PETSc.ScalarType)
    omega_coords = np.logical_or(
        (x[0] <= 0.2),
        np.logical_or(
            (x[0] >= 0.8),
            np.logical_or((x[1] >= 0.8), (x[1] <= 0.2)),
        ),
    )
    rest_coords = np.invert(omega_coords)
    values[omega_coords] = np.full(sum(omega_coords), 1.0)
    values[rest_coords] = np.full(sum(rest_coords), 0)
    return values


def data_domain_indicator_function_3d(x, data_size=0.25):
    values = np.zeros(x.shape[1], dtype=PETSc.ScalarType)
    omega_coords = np.logical_and(x[0] <= data_size, x[0] >= 0.0)
    rest_coords = np.invert(omega_coords)
    values[omega_coords] = np.full(sum(omega_coords), 1.0)
    values[rest_coords] = np.full(sum(rest_coords), 0)
    return values


def data_domain_indicator_function_3d_gcc(x, data_size=0.25):
    values = np.zeros(x.shape[1], dtype=PETSc.ScalarType)
    omega_coords = np.logical_or(
        (x[0] <= data_size),
        np.logical_or(
            (x[0] >= 1.0 - data_size),
            np.logical_or(
                (x[1] >= 1.0 - data_size),
                np.logical_or(
                    (x[1] <= data_size),
                    np.logical_or(
                        (x[2] <= data_size),
                        (x[2] >= 1.0 - data_size),
                    ),
                ),
            ),
        ),
    )
    rest_coords = np.invert(omega_coords)
    values[omega_coords] = np.full(sum(omega_coords), 1.0)
    values[rest_coords] = np.full(sum(rest_coords), 0)
    return values


def get_data_domain_indicator_expression_3d(mesh, data_size=0.25):
    x = ufl.SpatialCoordinate(mesh)
    indicator_condition = ufl.And(x[0] <= data_size, x[0] >= 0.0)
    return ufl.conditional(indicator_condition, 1, 0)


def get_data_domain_indicator_expression_3d_gcc(mesh, data_size=0.25):
    x = ufl.SpatialCoordinate(mesh)
    indicator_condition = ufl.Not(
        ufl.And(
            ufl.And(x[0] >= data_size, x[0] <= 1.0 - data_size),
            ufl.And(
                ufl.And(x[1] >= data_size, x[1] <= 1.0 - data_size),
                ufl.And(x[2] >= data_size, x[2] <= 1.0 - data_size),
            ),
        ),
    )
    return ufl.conditional(indicator_condition, 1, 0)


def sample_solution_1d(t, xu):
    return ufl.cos(2 * pi * (t)) * ufl.sin(2 * pi * xu[0])


def sample_solution_time_derivative_1d(t, xu):
    return -2 * pi * ufl.sin(2 * pi * (t)) * ufl.sin(2 * pi * xu[0])


def sample_solution_2d(t, xu):
    return ufl.cos(sqrt(2) * pi * t) * ufl.sin(pi * xu[0]) * ufl.sin(pi * xu[1])


def sample_solution_time_derivative_2d(t, xu):
    return (
        -sqrt(2)
        * pi
        * ufl.sin(sqrt(2) * pi * t)
        * ufl.sin(pi * xu[0])
        * ufl.sin(pi * xu[1])
    )


def sample_solution_3d(t, xu):
    return (
        ufl.cos(sqrt(3) * pi * t)
        * ufl.sin(pi * xu[0])
        * ufl.sin(pi * xu[1])
        * ufl.sin(pi * xu[2])
    )


def sample_solution_time_derivative_3d(t, xu):
    return (
        -sqrt(3)
        * pi
        * ufl.sin(sqrt(3) * pi * t)
        * ufl.sin(pi * xu[0])
        * ufl.sin(pi * xu[1])
        * ufl.sin(pi * xu[2])
    )


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
    mesh: Mesh,
    true_solution: ValueAndDerivative,
    data_domain_indicator: DataDomainIndicator,
    measure_errors: bool = True,
    plot_errors: bool = False,
    n_time_steps: int = 32,
    time_interval: float = 1.0,
    order: int = 1,
    solver_type: SolverType | None = None,
) -> None:
    if solver_type is None:
        solver_type = SolverType("gmres", "petsc" if pypardiso is None else "pypardiso")
    space_time_solver = SpaceTime(
        OrderTime(q=order, qstar=order if order == 1 else 0),
        OrderSpace(k=order, kstar=order),
        N=n_time_steps,
        T=time_interval,
        t=0.0,
        msh=mesh,
        omega=DataDomain(indicator_function=data_domain_indicator),
        stabilisation_terms={
            "data": 1e4,
            "dual": 1.0,
            "primal": 1e-3,
            "primal-jump": 1.0,
        },
        solution=true_solution,
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


def get_solve_problem_default_kwargs(
    dimension: int = 1,
    n_time_steps: int = -1,
    order: int = -1,
    time_interval: float = -1.0,
    use_unit_square_mesh_2d: bool = False,
    refinement_level: int = 3,
) -> dict:
    if dimension == 1:
        n_time_steps = 32 if n_time_steps <= 0 else n_time_steps
        return {
            "n_time_steps": n_time_steps,
            "order": 1 if order <= 0 else order,
            "time_interval": 1.0 if time_interval < 0 else time_interval,
            "mesh": get_mesh_1d(n_cells=5 * n_time_steps),
            "true_solution": ValueAndDerivative(
                sample_solution_1d,
                sample_solution_time_derivative_1d,
            ),
            "data_domain_indicator": data_domain_indicator_function_1d_convex,
        }
    elif dimension == 2:
        if use_unit_square_mesh_2d:
            n_time_steps = 8 if n_time_steps <= 0 else n_time_steps
            mesh = get_mesh_2d(n_cells=2 * n_time_steps)
        else:
            n_time_steps = 2**refinement_level if n_time_steps <= 0 else n_time_steps
            mesh = get_mesh_data_all_around(refinement_level, init_h_scale=5.0)[
                refinement_level
            ]
        return {
            "n_time_steps": n_time_steps,
            "order": 3 if args.order <= 0 else args.order,
            "time_interval": 1.0 if time_interval < 0 else time_interval,
            "mesh": mesh,
            "true_solution": ValueAndDerivative(
                sample_solution_2d,
                sample_solution_time_derivative_2d,
            ),
            "data_domain_indicator": data_domain_indicator_function_2d_convex,
        }
    elif dimension == 3:
        n_time_steps = 8 if n_time_steps <= 0 else n_time_steps
        n_cells = 2 * n_time_steps
        mesh = get_mesh_3d(n_cells=n_cells)
        data_domain_indicator = (
            data_domain_indicator_function_3d
            if n_cells > 2
            else get_data_domain_indicator_expression_3d(mesh)
        )
        return {
            "n_time_steps": n_time_steps,
            "order": 1 if order <= 0 else order,
            "time_interval": 0.5 if time_interval < 0 else time_interval,
            "mesh": mesh,
            "true_solution": ValueAndDerivative(
                sample_solution_3d,
                sample_solution_time_derivative_3d,
            ),
            "data_domain_indicator": data_domain_indicator,
        }
    else:
        msg = f"Unsupported dimension: {dimension}"
        raise ValueError(msg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Solve space-time inverse problem",
    )
    parser.add_argument(
        "--dimension",
        "-d",
        type=int,
        default=1,
        choices=(1, 2, 3),
        help="Dimension of spatial domain.",
    )
    parser.add_argument(
        "--n-time-steps",
        "-n",
        type=int,
        default=-1,
        help="Number of time steps. If negative, dimension specific default used.",
    )
    parser.add_argument(
        "--order",
        "-o",
        type=int,
        default=-1,
        help="Polynomial order. If negative, dimension specific default used.",
    )
    parser.add_argument(
        "--time-interval",
        "-t",
        type=float,
        default=-1.0,
        help="Size of time interval. If negative, dimension specific default used.",
    )
    args = parser.parse_args()
    start_time = time.time()
    solve_problem(**get_solve_problem_default_kwargs(**vars(args)))
    elapsed_time_seconds = time.time() - start_time
    memory_usage_bytes = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(f"Elapsed time  {elapsed_time_seconds} seconds")
    print(f"Memory usage in (GB) = {memory_usage_bytes / 1e6}")
