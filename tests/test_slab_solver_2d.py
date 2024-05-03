import sys
import time

import numpy as np
import pytest
import scipy.sparse as sp
from dolfinx import fem

from dxss._solvers import PySolver, get_lu_solver
from dxss.meshes import get_mesh_data_all_around
from dxss.solve_2d import REF_LVL_TO_N, dt_sample_sol, omega_ind_convex, sample_sol
from dxss.space_time import (
    DataDomain,
    OrderSpace,
    OrderTime,
    SpaceTime,
    ValueAndDerivative,
)

sys.setrecursionlimit(10**6)
SOLVER_TYPE = "petsc-LU"  # TODO: check with Janosch that this is sensible.


# define alternative solvers here
def get_sparse_matrix(mat):
    ai, aj, av = mat.getValuesCSR()
    return sp.csr_matrix((av, aj, ai))


REF_LVL = 1  # set to 1, coarsest mesh is fine for CI tests

t0 = 0
T = 1.0
N = REF_LVL_TO_N[REF_LVL]
ORDER = 2
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
LS_MESH = get_mesh_data_all_around(5, init_h_scale=5.0)
MSH = LS_MESH[REF_LVL]


ST = SpaceTime(
    OrderTime(q, qstar),
    OrderSpace(k, kstar),
    N=N,
    T=T,
    t=t0,
    msh=MSH,
    omega=DataDomain(indicator_function=omega_ind_convex),
    stabilisation_terms=STABS,
    solution=ValueAndDerivative(sample_sol, dt_sample_sol),
)
ST.setup_spacetime_finite_elements()
ST.prepare_precondition_gmres()

# matrix for linear sytems on the slabs
SLAB_MATRIX = ST.get_slab_matrix()


def test_slab_problem():
    # generate solution
    x_in, x_out = SLAB_MATRIX.createVecs()
    x_comp, _ = SLAB_MATRIX.createVecs()
    x_in.array[:] = np.random.default_rng().random(len(x_in.array))
    SLAB_MATRIX.mult(x_in, x_out)

    # Solvers for problems on the slabs

    start = time.time()

    if SOLVER_TYPE == "pypardiso":
        pardiso_solver = PySolver(get_sparse_matrix(SLAB_MATRIX))
        pardiso_solver.solve(x_out, x_comp, set_phase=False)
    elif SOLVER_TYPE == "petsc-LU":
        solver_slab = get_lu_solver(ST.msh, ST.get_slab_matrix())  # LU-decomposition
        solver_slab.solve(x_out, x_comp)
    else:
        msg = "invalid solver_type"
        raise ValueError(msg)

    end = time.time()
    error = np.linalg.norm(x_comp.array - x_in.array)
    print("Error = ", error)
    print("elapsed time  " + str(end - start) + " seconds")
    assert error < 1e-4


@pytest.mark.skip(reason=f"The error large with order {ORDER} and ref_lvl {REF_LVL}.")
def test_spacetime_solve():
    x_in = fem.petsc.create_vector(ST.SpaceTimeLfi)
    x_out = fem.petsc.create_vector(ST.SpaceTimeLfi)
    x_comp = fem.petsc.create_vector(ST.SpaceTimeLfi)
    x_in.array[:] = np.random.default_rng().random(len(x_in.array))
    ST.apply_spacetime_matrix(x_in, x_out)

    start = time.time()

    if SOLVER_TYPE == "pypardiso":
        ST.set_solver_slab(
            PySolver(get_sparse_matrix(ST.get_slab_matrix())),
        )  # general slab
        ST.set_solver_first_slab(
            PySolver(get_sparse_matrix(ST.get_slab_matrix_first_slab())),
        )
    elif SOLVER_TYPE == "petsc-LU":
        ST.set_solver_slab(get_lu_solver(ST.msh, ST.get_slab_matrix()))  # general slab
        ST.set_solver_first_slab(
            get_lu_solver(ST.msh, ST.get_slab_matrix_first_slab()),
        )  # first slab is special
    else:
        msg = "invalid solver_type"
        raise ValueError(msg)

    ST.pre_time_marching(x_out, x_comp)
    end = time.time()
    error = np.linalg.norm(x_comp.array - x_in.array)
    print("Error = ", error)
    print("elapsed time  " + str(end - start) + " seconds")

    assert error < 1e-4
