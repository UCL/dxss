import sys
import warnings
from math import pi, sqrt

import numpy as np
import ufl
from dolfinx.mesh import CellType, GhostMode, create_box
from mpi4py import MPI
from petsc4py import PETSc

from dxss.gmres import get_gmres_solution
from dxss.space_time import (
    DataDomain,
    OrderSpace,
    OrderTime,
    ProblemParameters,
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
GCC = False


def get_lu_solver(msh, mat):
    solver = PETSc.KSP().create(msh.comm)
    solver.setOperators(mat)
    solver.setType(PETSc.KSP.Type.PREONLY)
    solver.getPC().setType(PETSc.PC.Type.LU)
    return solver


class PySolver:
    def __init__(self, Asp, psolver):  # noqa: N803
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


REF_LVL_TO_N = [1, 2, 4, 8, 16, 32]
REF_LVL = 3

DATA_SIZE = 0.25

t0 = 0
T = 1 / 2
N = REF_LVL_TO_N[REF_LVL]
ORDER = 1
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

Nxs = [2, 4, 8, 16, 32, 64]

Nx = Nxs[REF_LVL]
print("Nx = ", Nx)
# define quantities depending on space
# for j in range(len(ls_mesh)):
#    with io.XDMFFile(ls_mesh[j].comm, "mesh-reflvl{0}.xdmf".format(j), "w") as xdmf:

MSH = create_box(
    MPI.COMM_WORLD,
    [np.array([0.0, 0.0, 0.0]), np.array([1.0, 1.0, 1.0])],
    [Nx, Nx, Nx],
    CellType.hexahedron,
    ghost_mode=GhostMode.shared_facet,
)


# with io.XDMFFile(msh.comm, "msh-hex-reflvl{0}.xdmf".format(ref_lvl), "w") as xdmf:
if GCC:
    if Nx > 2:

        def omega_ind(x):
            values = np.zeros(x.shape[1], dtype=PETSc.ScalarType)
            omega_coords = np.logical_or(
                (x[0] <= DATA_SIZE),
                np.logical_or(
                    (x[0] >= 1.0 - DATA_SIZE),
                    np.logical_or(
                        (x[1] >= 1.0 - DATA_SIZE),
                        np.logical_or(
                            (x[1] <= DATA_SIZE),
                            np.logical_or(
                                (x[2] <= DATA_SIZE),
                                (x[2] >= 1.0 - DATA_SIZE),
                            ),
                        ),
                    ),
                ),
            )
            rest_coords = np.invert(omega_coords)
            values[omega_coords] = np.full(sum(omega_coords), 1.0)
            values[rest_coords] = np.full(sum(rest_coords), 0)
            return values

    else:
        x = ufl.SpatialCoordinate(MSH)
        omega_indicator = ufl.Not(
            ufl.And(
                ufl.And(x[0] >= DATA_SIZE, x[0] <= 1.0 - DATA_SIZE),
                ufl.And(
                    ufl.And(x[1] >= DATA_SIZE, x[1] <= 1.0 - DATA_SIZE),
                    ufl.And(x[2] >= DATA_SIZE, x[2] <= 1.0 - DATA_SIZE),
                ),
            ),
        )
        omega_ind = ufl.conditional(omega_indicator, 1, 0)
elif Nx > 2:

    def omega_ind(x):
        values = np.zeros(x.shape[1], dtype=PETSc.ScalarType)
        omega_coords = np.logical_and((x[0] <= DATA_SIZE), (x[0] >= 0.0))
        rest_coords = np.invert(omega_coords)
        values[omega_coords] = np.full(sum(omega_coords), 1.0)
        values[rest_coords] = np.full(sum(rest_coords), 0)
        return values

else:
    x = ufl.SpatialCoordinate(MSH)
    omega_indicator = ufl.And(x[0] <= DATA_SIZE, x[0] >= 0.0)
    omega_ind = ufl.conditional(omega_indicator, 1, 0)


def sample_sol(t, xu):
    return (
        ufl.cos(sqrt(3) * pi * t)
        * ufl.sin(pi * xu[0])
        * ufl.sin(pi * xu[1])
        * ufl.sin(pi * xu[2])
    )


def dt_sample_sol(t, xu):
    return (
        -sqrt(3)
        * pi
        * ufl.sin(sqrt(3) * pi * t)
        * ufl.sin(pi * xu[0])
        * ufl.sin(pi * xu[1])
        * ufl.sin(pi * xu[2])
    )


ST = SpaceTime(
    OrderTime(q, qstar),
    OrderSpace(k, kstar),
    N=N,
    T=T,
    t=t0,
    msh=MSH,
    omega=DataDomain(indicator_function=omega_ind, fitted=Nx > 2),
    stabilisation_terms=STABS,
    solution=ValueAndDerivative(sample_sol, dt_sample_sol),
    parameters=ProblemParameters(),
)
ST.setup_spacetime_finite_elements()
ST.prepare_precondition_gmres()
A_space_time_linop = ST.get_spacetime_matrix_as_linear_operator()
b_rhs = ST.get_spacetime_rhs()

# Prepare the solvers for problems on the slabs


def solve_problem(measure_errors=False):
    start = time.time()
    if SOLVER_TYPE == "pypardiso":
        genreal_slab_solver = pypardiso.PyPardisoSolver()
        slab_matrix_sparse = get_sparse_matrix(ST.get_slab_matrix())

        genreal_slab_solver.factorize(slab_matrix_sparse)
        ST.set_solver_slab(PySolver(slab_matrix_sparse, genreal_slab_solver))

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
            pre=ST.pre_time_marching_improved,
            maxsteps=100000,
            tol=1e-7,
            printrates=True,
        )

    elif SOLVER_TYPE == "petsc-LU":
        ST.set_solver_slab(get_lu_solver(ST.msh, ST.get_slab_matrix()))  # general slab
        ST.set_solver_first_slab(
            get_lu_solver(ST.msh, ST.get_slab_matrix_first_slab()),
        )  # first slab is special
        u_sol, res = get_gmres_solution(
            A_space_time_linop,
            b_rhs,
            pre=ST.pre_time_marching_improved,
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
        ST.plot_para_view(u_sol, name=f"abserr-cube-GCC-order{ORDER}")
        ST.measured_errors(u_sol)


if __name__ == "__main__":
    solve_problem(measure_errors=True)

    print(
        "Memory usage in (Gb) = ",
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6,
    )
