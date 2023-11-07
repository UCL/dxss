import sys
import warnings
from math import pi

import numpy as np
import ufl
from dolfinx.mesh import create_unit_interval
from mpi4py import MPI
from petsc4py import PETSc

from dxss.gmres import get_gmres_solution
from dxss.space_time import (
    DataDomain,
    OrderSpace,
    OrderTime,
    SpaceTime,
    ValueAndDerivative,
    get_sparse_matrix,
)

try:
    import pypardiso

    SOLVER_TYPE = "pypardiso"
except ImportError:
    pypardiso = None
    SOLVER_TYPE = "petsc-LU"
import resource
import time

sys.setrecursionlimit(10**6)


def get_lu_solver(msh, mat):
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
        self.solver._check_A(self.Asp)
        b = self.solver._check_b(self.Asp, b_inp.array)
        self.solver.set_phase(33)
        x_out.array[:] = self.solver._call_pardiso(self.Asp, b)[:]


t0 = 0
T = 1
N = 32
N_x = int(5 * N)
ORDER = 1
k = ORDER
q = ORDER
kstar = 1
qstar = 1 if ORDER == 1 else 0

STABS = {
    "data": 1e4,
    "dual": 1.0,
    "primal": 1e-3,
    "primal-jump": 1.0,
}

# define quantities depending on space
MSH = create_unit_interval(MPI.COMM_WORLD, N_x)


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


ST = SpaceTime(
    OrderTime(q, qstar),
    OrderSpace(k, kstar),
    N=N,
    T=T,
    t=t0,
    msh=MSH,
    omega=DataDomain(indicator_function=omega_ind_nogcc),
    stabilisation_terms=STABS,
    solution=ValueAndDerivative(sample_sol, dt_sample_sol),
)
ST.setup_spacetime_finite_elements()
ST.prepare_precondition_gmres()
A_space_time_linop = ST.get_spacetime_matrix_as_linear_operator()
b_rhs = ST.get_spacetime_rhs()
# Prepare the solvers for problems on the slabs
# GMRes iteration


# Prepare coarse grid correction


def solve_problem(measure_errors=False):
    start = time.time()
    if SOLVER_TYPE == "pypardiso":
        genreal_slab_solver = pypardiso.PyPardisoSolver()
        sparse_slab_matrix = get_sparse_matrix(ST.get_slab_matrix())

        genreal_slab_solver.factorize(sparse_slab_matrix)
        ST.set_solver_slab(PySolver(sparse_slab_matrix, genreal_slab_solver))

        initial_slab_solver = pypardiso.PyPardisoSolver()
        slab_matrix_first_slab_sparse = get_sparse_matrix(
            ST.get_slab_matrix_first_slab(),
        )
        initial_slab_solver.factorize(slab_matrix_first_slab_sparse)
        ST.set_solver_first_slab(
            PySolver(slab_matrix_first_slab_sparse, initial_slab_solver),
        )

        u_sol, res = get_gmres_solution(
            A_space_time_linop,
            b_rhs,
            maxsteps=100000,
            tol=1e-7,
            printrates=True,
        )

        ST.plot_error(u_sol, n_space=500, n_time_subdiv=20)
    elif SOLVER_TYPE == "petsc-LU":
        ST.set_solver_slab(get_lu_solver(ST.msh, ST.get_slab_matrix()))  # general slab
        ST.set_solver_first_slab(
            get_lu_solver(ST.msh, ST.get_slab_matrix_first_slab()),
        )  # first slab is special
        u_sol, res = get_gmres_solution(
            A_space_time_linop,
            b_rhs,
            maxsteps=100000,
            tol=1e-7,
            printrates=True,
        )
    else:
        A_space_time = ST.get_spacetime_matrix()  # noqa: N806
        u_sol, _ = A_space_time.createVecs()
        solver_space_time = get_lu_solver(ST.msh, A_space_time)
        solver_space_time.solve(u_sol, b_rhs)

    end = time.time()
    print("elapsed time  " + str(end - start) + " seconds")

    if measure_errors:
        ST.measured_errors(u_sol)


if __name__ == "__main__":
    solve_problem(measure_errors=True)

    print(
        "Memory usage in (Gb) = ",
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6,
    )
