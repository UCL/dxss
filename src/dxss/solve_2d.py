import resource
import sys
import time
from math import pi, sqrt

import numpy as np
import ufl
from petsc4py import PETSc

import dxss._solver_backend
from dxss._solvers import PySolver, get_lu_solver
from dxss.gmres import get_gmres_solution
from dxss.meshes import get_mesh_data_all_around
from dxss.space_time import (
    DataDomain,
    OrderSpace,
    OrderTime,
    SpaceTime,
    SpaceTimePETScMatrixWrapper,
    SpaceTimePETScPreconditionerWrapper,
    ValueAndDerivative,
    get_sparse_matrix,
)

sys.setrecursionlimit(10**6)


REF_LVL_TO_N = [1, 2, 4, 8, 16, 32]
REF_LVL = 3

t0 = 0
T = 1.0
N = REF_LVL_TO_N[REF_LVL]
ORDER = 3
k = ORDER
q = ORDER
kstar = ORDER
qstar = ORDER
STABS = {
    "data": 1e4,
    "dual": 1.0,
    "primal": 1e-3,
    "primal-jump": 1.0,
}

# define quantities depending on space
ls_mesh = get_mesh_data_all_around(5, init_h_scale=5.0)
# for j in range(len(ls_mesh)):
#    with io.XDMFFile(ls_mesh[j].comm, "mesh-reflvl{0}.xdmf".format(j), "w") as xdmf:
msh = ls_mesh[REF_LVL]


def omega_ind_convex(x):
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


def sample_sol(t, xu):
    return ufl.cos(sqrt(2) * pi * t) * ufl.sin(pi * xu[0]) * ufl.sin(pi * xu[1])


def dt_sample_sol(t, xu):
    return (
        -sqrt(2)
        * pi
        * ufl.sin(sqrt(2) * pi * t)
        * ufl.sin(pi * xu[0])
        * ufl.sin(pi * xu[1])
    )


ST = SpaceTime(
    OrderTime(q, qstar),
    OrderSpace(k, kstar),
    N=N,
    T=T,
    t=t0,
    msh=msh,
    omega=DataDomain(indicator_function=omega_ind_convex),
    stabilisation_terms=STABS,
    solution=ValueAndDerivative(sample_sol, dt_sample_sol),
)
ST.setup_spacetime_finite_elements()
ST.prepare_precondition_gmres()
b_rhs = ST.get_spacetime_rhs()

# Prepare the solvers for problems on the slabs


def solve_problem(measure_errors=False):
    start = time.time()
    if dxss._solver_backend.SOLVER_TYPE == "pypardiso":
        genreal_slab_solver = dxss._solver_backend.pypardiso.PyPardisoSolver()
        slab_matrix_sparse = get_sparse_matrix(ST.get_slab_matrix())

        genreal_slab_solver.factorize(slab_matrix_sparse)
        ST.set_solver_slab(PySolver(slab_matrix_sparse, genreal_slab_solver))

        initial_slab_solver = dxss._solver_backend.pypardiso.PyPardisoSolver()
        slab_matrix_first_slab_sparse = get_sparse_matrix(
            ST.get_slab_matrix_first_slab(),
        )
        initial_slab_solver.factorize(slab_matrix_first_slab_sparse)
        ST.set_solver_first_slab(
            PySolver(slab_matrix_first_slab_sparse, initial_slab_solver),
        )

        u_sol, _ = get_gmres_solution(
            A=SpaceTimePETScMatrixWrapper(ST),
            b=b_rhs,
            pre=SpaceTimePETScPreconditionerWrapper(ST),
            maxsteps=100000,
            tol=1e-7,
            printrates=True,
        )
    elif dxss._solver_backend.SOLVER_TYPE == "petsc-LU":
        ST.set_solver_slab(get_lu_solver(ST.msh, ST.get_slab_matrix()))  # general slab
        ST.set_solver_first_slab(
            get_lu_solver(ST.msh, ST.get_slab_matrix_first_slab()),
        )  # first slab is special
        u_sol, _ = get_gmres_solution(
            A=SpaceTimePETScMatrixWrapper(ST),
            b=b_rhs,
            pre=SpaceTimePETScPreconditionerWrapper(ST),
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
