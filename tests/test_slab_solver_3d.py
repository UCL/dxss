import sys
import time

import numpy as np
import scipy.sparse as sp
from dolfinx.mesh import CellType, GhostMode, create_box
from mpi4py import MPI
from petsc4py import PETSc

from dxss._solvers import PySolver, get_lu_solver
from dxss.solve_3d import dt_sample_sol, sample_sol
from dxss.space_time import (
    DataDomain,
    OrderSpace,
    OrderTime,
    SpaceTime,
    ValueAndDerivative,
)

try:
    import pypardiso

    SOLVER_TYPE = "pypardiso"
except ImportError:
    SOLVER_TYPE = "petsc-LU"

sys.setrecursionlimit(10**6)
SOLVER_TYPE = "petsc-LU"


# define alternative solvers here
def get_sparse_matrix(mat):
    ai, aj, av = mat.getValuesCSR()
    return sp.csr_matrix((av, aj, ai))


REF_LVL_TO_N = [1, 2, 4, 8, 16, 32]
REF_LVL = 1
Nxs = [2, 4, 8, 16, 32, 64]
Nx = Nxs[REF_LVL]

DATA_SIZE = 0.25
t0 = 0
T = 1.0
N = REF_LVL_TO_N[REF_LVL]
ORDER = 3
k = ORDER
q = ORDER
kstar = 1
qstar = 0
STABS = {
    "data": 1e4,
    "dual": 1.0,
    "primal": 1e-3,
    "primal-jump": 1.0,
}

# define quantities depending on space
MSH = create_box(
    MPI.COMM_WORLD,
    [np.array([0.0, 0.0, 0.0]), np.array([1.0, 1.0, 1.0])],
    [Nx, Nx, Nx],
    CellType.hexahedron,
    ghost_mode=GhostMode.shared_facet,
)


def omega_ind_convex(x):
    values = np.zeros(x.shape[1], dtype=PETSc.ScalarType)
    omega_coords = np.logical_or(
        (x[0] <= DATA_SIZE),
        np.logical_or(
            (x[0] >= 1.0 - DATA_SIZE),
            np.logical_or(
                (x[1] >= 1.0 - DATA_SIZE),
                np.logical_or(
                    (x[1] <= DATA_SIZE),
                    np.logical_or((x[2] <= DATA_SIZE), (x[2] >= 1.0 - DATA_SIZE)),
                ),
            ),
        ),
    )
    rest_coords = np.invert(omega_coords)
    values[omega_coords] = np.full(sum(omega_coords), 1.0)
    values[rest_coords] = np.full(sum(rest_coords), 0)
    return values


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


def test_slab_problem():
    # matrix for linear systems on the slabs
    slab_matrix = ST.get_slab_matrix()

    # generate solution
    x_in, x_out = slab_matrix.createVecs()
    x_comp, _ = slab_matrix.createVecs()
    x_in.array[:] = np.random.default_rng().random(len(x_in.array))
    slab_matrix.mult(x_in, x_out)

    start = time.time()

    if SOLVER_TYPE == "pypardiso":
        pardiso_solver = PySolver(get_sparse_matrix(slab_matrix))
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
    print("elapsed time = " + str(end - start) + " seconds")

    assert error < 1e-4
