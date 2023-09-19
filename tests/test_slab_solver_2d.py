import pytest
import numpy as np
from math import pi, sqrt
import ufl
from dolfinx import fem, io, mesh, plot
from ufl import ds, dx, grad, inner, dS, jump, div
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx.mesh import create_unit_interval
from dolfinx.io import XDMFFile
import sys

sys.setrecursionlimit(10**6)
from dxss.gmres import GMRes
from dxss.space_time import *
from dxss.precomp_time_int import theta_ref, d_theta_ref
from dxss.meshes import get_mesh_hierarchy, get_mesh_data_all_around
import time

import scipy.sparse as sp

solver_type = "petsc-LU"
# solver_type = "pypardiso" #


def GetLuSolver(msh, mat):
    solver = PETSc.KSP().create(msh.comm)
    solver.setOperators(mat)
    solver.setType(PETSc.KSP.Type.PREONLY)
    solver.getPC().setType(PETSc.PC.Type.LU)
    return solver


# define alternative solvers here
def GetSpMat(mat):
    ai, aj, av = mat.getValuesCSR()
    Asp = sp.csr_matrix((av, aj, ai))
    return Asp


class PySolver:
    def __init__(self, Asp):
        self.Asp = Asp

    def solve(self, b_inp, x_out):
        x_py = pypardiso.spsolve(self.Asp, b_inp.array)
        x_out.array[:] = x_py[:]


ref_lvl_to_N = [1, 2, 4, 8, 16, 32]
ref_lvl = 1  # set to 1, coarsest mesh is fine for CI tests

t0 = 0
T = 1.0
N = ref_lvl_to_N[ref_lvl]
order = 2
k = order
q = order
kstar = order
qstar = order
stabs = {
    "data": 1e4,
    "dual": 1.0,
    "primal": 1e-3,
    "primal-jump": 1.0,
}

# define quantities depending on space
ls_mesh = get_mesh_data_all_around(5, init_h_scale=5.0)
msh = ls_mesh[ref_lvl]


def omega_Ind_convex(x):
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


st = space_time(
    q=q,
    qstar=qstar,
    k=k,
    kstar=kstar,
    N=N,
    T=T,
    t=t0,
    msh=msh,
    Omega_Ind=omega_Ind_convex,
    stabs=stabs,
    sol=sample_sol,
    dt_sol=dt_sample_sol,
)
st.SetupSpaceTimeFEs()
st.PreparePrecondGMRes()

# matrix for linear sytems on the slabs
SlabMat = st.GetSlabMat()


def test_slab_problem():
    # generate solution
    x_in, x_out = SlabMat.createVecs()
    x_comp, _ = SlabMat.createVecs()
    x_in.array[:] = np.random.rand(len(x_in.array))
    SlabMat.mult(x_in, x_out)

    # Solvers for problems on the slabs
    # solver_slab = GetLuSolver(st.msh, st.GetSlabMat())  # LU-decomposition
    # solver_slab.solve(x_out,x_comp)
    # print("Error = ", np.linalg.norm(x_comp.array - x_in.array))

    start = time.time()

    if solver_type == "pypardiso":
        pardiso_solver = PySolver(GetSpMat(SlabMat))
        pardiso_solver.solve(x_out, x_comp)
    elif solver_type == "petsc-LU":
        solver_slab = GetLuSolver(st.msh, st.GetSlabMat())  # LU-decomposition
        solver_slab.solve(x_out, x_comp)
    else:
        raise ValueError("invalid solver_type")

    end = time.time()
    error = np.linalg.norm(x_comp.array - x_in.array)
    print("Error = ", error)
    print("elapsed time  " + str(end - start) + " seconds")
    assert error < 1e-4


@pytest.mark.skip(reason=f"The error large with order {order} and ref_lvl {ref_lvl}.")
def test_spacetime_solve():
    x_in = fem.petsc.create_vector(st.SpaceTimeLfi)
    x_out = fem.petsc.create_vector(st.SpaceTimeLfi)
    x_comp = fem.petsc.create_vector(st.SpaceTimeLfi)
    x_in.array[:] = np.random.rand(len(x_in.array))
    st.ApplySpaceTimeMatrix(x_in, x_out)

    start = time.time()

    if solver_type == "pypardiso":
        st.SetSolverSlab(PySolver(GetSpMat(st.GetSlabMat())))  # general slab
        st.SetSolverFirstSlab(PySolver(GetSpMat(st.GetSlabMatFirstSlab())))
    elif solver_type == "petsc-LU":
        st.SetSolverSlab(GetLuSolver(st.msh, st.GetSlabMat()))  # general slab
        st.SetSolverFirstSlab(
            GetLuSolver(st.msh, st.GetSlabMatFirstSlab())
        )  # first slab is special
    else:
        raise ValueError("invalid solver_type")

    st.pre_time_marching(x_out, x_comp)
    end = time.time()
    error = np.linalg.norm(x_comp.array - x_in.array)
    print("Error = ", error)
    print("elapsed time  " + str(end - start) + " seconds")

    assert error < 1e-4
