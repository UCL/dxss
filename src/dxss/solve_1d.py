import sys
from math import pi

import numpy as np
import ufl
from dolfinx.mesh import create_unit_interval
from mpi4py import MPI
from petsc4py import PETSc

sys.setrecursionlimit(10**6)
from dxss.gmres import GMRes
from dxss.space_time import *

try:
    import pypardiso

    solver_type = "pypardiso"  #
except ImportError:
    solver_type = "petsc-LU"
import resource
import time


def GetLuSolver(msh, mat):
    solver = PETSc.KSP().create(msh.comm)
    solver.setOperators(mat)
    solver.setType(PETSc.KSP.Type.PREONLY)
    solver.getPC().setType(PETSc.PC.Type.LU)
    return solver


class PySolver:
    def __init__(self, Asp, psolver):
        self.Asp = Asp
        self.solver = psolver

    def solve(self, b_inp, x_out):
        self.solver._check_A(self.Asp)
        b = self.solver._check_b(self.Asp, b_inp.array)
        self.solver.set_phase(33)
        x_out.array[:] = self.solver._call_pardiso(self.Asp, b)[:]


t0 = 0
T = 1
N = 32
N_x = int(5 * N)
order = 1
k = order
q = order
kstar = 1
qstar = 1 if order == 1 else 0

stabs = {
    "data": 1e4,
    "dual": 1.0,
    "primal": 1e-3,
    "primal-jump": 1.0,
}

# stabs = {"data": 1e0,
#        "primal-jump":1.0,

# define quantities depending on space
msh = create_unit_interval(MPI.COMM_WORLD, N_x)


def omega_Ind_convex(x):
    values = np.zeros(x.shape[1], dtype=PETSc.ScalarType)
    omega_coords = np.logical_or((x[0] <= 0.2), (x[0] >= 0.8))
    rest_coords = np.invert(omega_coords)
    values[omega_coords] = np.full(sum(omega_coords), 1.0)
    values[rest_coords] = np.full(sum(rest_coords), 0)
    return values


def omega_Ind_noGCC(x):
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


# def sample_sol(t,xu):

# def dt_sample_sol(t,xu):


st = space_time(
    q=q,
    qstar=qstar,
    k=k,
    kstar=kstar,
    N=N,
    T=T,
    t=t0,
    msh=msh,
    Omega_Ind=omega_Ind_noGCC,
    stabs=stabs,
    sol=sample_sol,
    dt_sol=dt_sample_sol,
)
st.SetupSpaceTimeFEs()
st.PreparePrecondGMRes()
A_space_time_linop = st.GetSpaceTimeMatrixAsLinearOperator()
b_rhs = st.GetSpaceTimeRhs()
# Prepare the solvers for problems on the slabs
# GMRes iteration


# Prepare coarse grid correction


def SolveProblem(measure_errors=False):
    start = time.time()
    if solver_type == "pypardiso":
        genreal_slab_solver = pypardiso.PyPardisoSolver()
        SlabMatSp = GetSpMat(st.GetSlabMat())

        genreal_slab_solver.factorize(SlabMatSp)
        st.SetSolverSlab(PySolver(SlabMatSp, genreal_slab_solver))

        initial_slab_solver = pypardiso.PyPardisoSolver()
        SlabMatFirstSlabSp = GetSpMat(st.GetSlabMatFirstSlab())
        initial_slab_solver.factorize(SlabMatFirstSlabSp)
        st.SetSolverFirstSlab(PySolver(SlabMatFirstSlabSp, initial_slab_solver))

        u_sol, res = GMRes(
            A_space_time_linop,
            b_rhs,
            pre=st.pre_time_marching_improved,
            maxsteps=100000,
            tol=1e-7,
            startiteration=0,
            printrates=True,
        )

        st.PlotError(u_sol, N_space=500, N_time_subdiv=20)
    elif solver_type == "petsc-LU":
        st.SetSolverSlab(GetLuSolver(st.msh, st.GetSlabMat()))  # general slab
        st.SetSolverFirstSlab(
            GetLuSolver(st.msh, st.GetSlabMatFirstSlab()),
        )  # first slab is special
        u_sol, res = GMRes(
            A_space_time_linop,
            b_rhs,
            pre=st.pre_time_marching_improved,
            maxsteps=100000,
            tol=1e-7,
            startiteration=0,
            printrates=True,
        )
    else:
        A_space_time = st.GetSpaceTimeMatrix()
        u_sol, _ = A_space_time.createVecs()
        solver_space_time = GetLuSolver(st.msh, A_space_time)
        solver_space_time.solve(u_sol, b_rhs)

    end = time.time()
    print("elapsed time  " + str(end - start) + " seconds")

    if measure_errors:
        st.MeasureErrors(u_sol)


if __name__ == "__main__":
    SolveProblem(measure_errors=True)

    print(
        "Memory usage in (Gb) = ",
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6,
    )
