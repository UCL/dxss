import sys
import time
from math import pi, sqrt

import numpy as np
import pytest
import scipy.sparse as sp
import ufl
from dolfinx import fem
from petsc4py import PETSc

from dxss.meshes import get_mesh_data_all_around
from dxss.space_time import SpaceTime

try:
    import pypardiso

    SOLVER_TYPE = "pypardiso"
except ImportError:
    SOLVER_TYPE = "petsc-LU"

sys.setrecursionlimit(10**6)
SOLVER_TYPE = "petsc-LU"


def get_lu_solver(msh, mat):
    solver = PETSc.KSP().create(msh.comm)
    solver.setOperators(mat)
    solver.setType(PETSc.KSP.Type.PREONLY)
    solver.getPC().setType(PETSc.PC.Type.LU)
    return solver


# define alternative solvers here
def get_sparse_matrix(mat):
    ai, aj, av = mat.getValuesCSR()
    return sp.csr_matrix((av, aj, ai))


class PySolver:
    def __init__(self, Asp):  # noqa: N803
        self.Asp = Asp

    def solve(self, b_inp, x_out):
        x_py = pypardiso.spsolve(self.Asp, b_inp.array)
        x_out.array[:] = x_py[:]


REF_LVL_TO_N = [1, 2, 4, 8, 16, 32]
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


def omega_ind_convex(x):
    values = np.zeros(x.shape[1], dtype=PETSc.ScalarType)
    omega_coords = np.logical_or(
        (x[0] <= 0.2),
        np.logical_or((x[0] >= 0.8), np.logical_or((x[1] >= 0.8), (x[1] <= 0.2))),
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
    q=q,
    qstar=qstar,
    k=k,
    kstar=kstar,
    N=N,
    T=T,
    t=t0,
    msh=MSH,
    omega_ind=omega_ind_convex,
    stabs=STABS,
    sol=sample_sol,
    dt_sol=dt_sample_sol,
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
        pardiso_solver.solve(x_out, x_comp)
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
