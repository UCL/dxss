from __future__ import annotations

import dataclasses
from math import sqrt
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
import ufl
from dolfinx import fem, io
from dolfinx.fem.petsc import assemble_matrix
from dxh import evaluate_function_at_points
from matplotlib import colors
from matplotlib.ticker import MaxNLocator
from mpi4py import MPI
from ufl import div, dS, ds, grad, inner, jump

from dxss.precomp_time_int import (
    QuadRule,
    basis_in_time,
    d_theta_ref,
    get_elmat_time,
    theta_ref,
)


@dataclasses.dataclass
class OrderTime:
    q: int
    qstar: int


@dataclasses.dataclass
class OrderSpace:
    k: int
    kstar: int


@dataclasses.dataclass
class ValueAndDerivative:
    v: np.ndarray
    dvdt: np.ndarray


@dataclasses.dataclass
class DataDomain:
    indicator_function: Callable | fem.Function = None
    fitted: bool = True


@dataclasses.dataclass
class ProblemParameters:
    jumps_in_fw_problem: bool = False
    well_posed: bool = False

    def __iter__(self):
        return (getattr(self, field.name) for field in dataclasses.fields(self))


_DEFAULT_PARAMETERS = ProblemParameters()


# TODO: this is duplicated... factor out to a "PETSc interface functions" file?
def get_sparse_matrix(mat):
    ai, aj, av = mat.getValuesCSR()
    return sp.csr_matrix((av, aj, ai))


class SpaceTime:
    # TODO:
    # this is a *large class* (https://refactoring.guru/smells/large-class) with
    # a *long parameter list* constructor
    # (https://refactoring.guru/smells/long-parameter-list)
    #
    # We can create intermediary data objects (where relevant) or split up some
    # of the configuration of this object.  Ideally we'd also split up the
    # object into smaller classes or functions.  For example do q and qstar
    # always get passed around together, Should they be a tuple? Ditto k and
    # kstar.
    #
    #  - PLR0913 there are far too many parameters being passed to the
    #            constructor.
    #  - PLR0915 there are too many conditional statements.
    def __init__(  # noqa: PLR0913, PLR0915
        self,
        polynomial_order_time: OrderTime,
        polynomial_order_space: OrderSpace,
        N,  # noqa: N803
        T,  # noqa: N803
        t,
        msh,
        stabilisation_terms: dict[str, float],
        omega: DataDomain,
        solution: ValueAndDerivative,
        parameters: ProblemParameters = _DEFAULT_PARAMETERS,
    ):
        """Construct a SpaceTime object.

        Args:
            polynomial_order_time: Polynomial degrees of the finite elements in space.
            polynomial_order_space: Polynomial degrees of the finite elements in time.
            stabilisation_terms: A dictionary of stabilisation terms.
            omega: A DataDomain object holding the indicator function.
            solution: A best starting guess for the solution and its derivative.
            parameters: Problem-specific parameters.
            ...
        """
        self.name = "space-time-wave"
        self.otime: OrderTime = polynomial_order_time
        self.ospace: OrderSpace = polynomial_order_space
        self.N = N
        self.T = T
        self.t = t
        self.delta_t = self.T / self.N
        self.msh = msh
        self.omega = omega
        self.x = ufl.SpatialCoordinate(msh)
        self.solution = solution
        self.lam_Nitsche = 5 * self.ospace.k**2
        self.stabilisation_terms = stabilisation_terms
        self.jumps_in_fw_problem, self.well_posed = parameters

        # mesh-related
        self.metadata = {"quadrature_degree": 2 * self.ospace.k + 3}
        self.dx = ufl.Measure("dx", domain=self.msh, metadata=self.metadata)
        self.n_facet = ufl.FacetNormal(self.msh)
        self.h = ufl.CellDiameter(self.msh)

        # derived quantities for time discretization
        self.elmat_time = get_elmat_time(self.otime.q, self.otime.qstar)
        self.qr = QuadRule("Gauss-Radau", 4)
        self.qr_ho = QuadRule("Gauss-Radau", 5)
        self.phi_trial, self.dt_phi_trial = basis_in_time[self.otime.q]
        self.phi_test, self.dt_phi_test = basis_in_time[self.otime.q]
        self.subspace_time_order = [
            self.otime.q,
            self.otime.q,
            self.otime.qstar,
            self.otime.qstar,
        ]

        # DG0 indicator function
        # todo: could this be moved to the constructor of the dataclass?
        if omega.fitted:
            q_ind = fem.FunctionSpace(self.msh, ("DG", 0))
            self.omega.indicator_function = fem.Function(q_ind)
            self.omega.indicator_function.interpolate(self.omega.indicator_function)
        else:
            self.omega.indicator_function = self.omega.indicator_function

        # FESpaces
        self.fes = None
        self.fes_u1_0 = None
        self.fes_u2_0 = None
        self.dofmaps_full = None
        self.fes_coupling = None
        self.fes_p = None
        self.fes_u1_0_pre = None
        self.fes_u2_0_pre = None
        self.fes_slab_bnd = None

        # Dofmaps
        self.dofmaps_full = None
        self.dofmaps_coupling = None
        self.dofmap_fes_u1_0_pre = None
        self.dofmap_fes_u2_0_pre = None
        self.dofmaps_slab_bnd = None

        # test functions for pre
        self.w1_p = None
        self.w2_p = None
        self.y1_p = None
        self.y2_p = None

        # Bilinear forms / matrices
        # SpaceTime
        self.spacetime_bilinear_form = None
        self.spacetime_matrix = None
        # Slab matrix with DG-jumps
        self.slab_bilinear_form = None
        self.slab_matrix = None
        self.solver_slab = None
        # Slab matrix without DG-jumps
        self.slab_bilinear_form_no_dg_jumps = None
        self.slab_matrix_no_dg_jumps = None
        self.solver_slab_no_dg_jumps = None

        # Coupling between slice
        self.coupling_matrix_between_slices = None
        # Scaled mass matrix between slice
        self.scaled_mass_matrix_between_slices = None

        # Special treatment of first time-slab
        self.SlabBfiFirstSlab = None
        self.SlabMatFirstSlab = None
        self.SolverFirstSlab = None

        # LinearForm
        self.SpaceTimeLfi = None
        self.SpaceTimeRhs = None

        # auxiliary vectors
        self.vec_slab1 = None
        self.vec_slab2 = None
        self.vec_coupling1 = None
        self.vec_coupling2 = None
        self.u1_0_pre = None
        self.u2_0_pre = None
        self.vec_0_bnd_in = None
        self.vec_0_bnd_out = None

        # auxiliary FEM functions
        self.u1_0 = None
        self.u2_0 = None
        self.uh_pre = None
        self.uh = None
        self.u1_minus = None
        self.u1_slab = None

        # Miscalleneous
        self.sample_pts_error = np.linspace(0, self.T, 105).tolist()
        self.eps: float = 0.0
        if self.otime.qstar == 0:
            self.eps = 1e-15  # hack to trick form compiler

        # for coarse grid correction
        self.msh_coarse = None
        self.fes_coarse = None
        self.dofmaps_full_coarse = None
        self.SpaceTimeMatCoarse = None
        self.SpaceTimeBfiCoarse = None
        self.SolverCoarse = None
        self.tmp1_fine = None
        self.tmp2_fine = None
        self.tmp3_sol = None
        self.tmp4_sol = None
        self.tmp5_sol = None
        self.tmp6_sol = None
        self.tmp1_coarse = None
        self.tmp2_coarse = None
        self.fes_node_coarse = None
        self.dofmap_node_coarse = None
        self.fes_node_fine = None
        self.dofmap_node_fine = None
        self.u_node_fine = None
        self.u_node_coarse = None

    def setup_spacetime_finite_elements(self):
        # Full Space-time system
        vel_q = ufl.VectorElement(
            "CG",
            self.msh.ufl_cell(),
            self.ospace.k,
            int((self.otime.q + 1) * self.N),
        )
        vel_qstar = ufl.VectorElement(
            "CG",
            self.msh.ufl_cell(),
            self.ospace.kstar,
            int((self.otime.qstar + 1) * self.N),
        )
        mel = ufl.MixedElement([vel_q, vel_q, vel_qstar, vel_qstar])
        self.fes = fem.FunctionSpace(self.msh, mel)
        dofmaps_full = []
        for j in range(4):
            dofmaps_j = []
            int_idx_q = 0
            int_idx_qstar = 0
            for _n in range(self.N):
                for ell in range(self.subspace_time_order[j] + 1):
                    shift_idx = int_idx_q if j in [0, 1] else int_idx_qstar
                    dofmaps_j.append(self.fes.sub(j).sub(shift_idx + ell).collapse()[1])
                int_idx_q += self.otime.q + 1
                int_idx_qstar += self.otime.qstar + 1
            dofmaps_full.append(dofmaps_j)
        self.dofmaps_full = dofmaps_full

        self.uh = fem.Function(self.fes)

        self.fes_u1_0 = self.fes.sub(0).sub(0).collapse()[0]
        self.fes_u2_0 = self.fes.sub(1).sub(0).collapse()[0]
        self.u1_0 = fem.Function(self.fes_u1_0)
        self.u2_0 = fem.Function(self.fes_u2_0)
        self.u1_minus = fem.Function(self.fes_u1_0)
        self.u1_slab = [fem.Function(self.fes_u1_0) for j in range(self.otime.q + 1)]

        # For strong coupling between slabs
        vel_coupling = ufl.VectorElement("CG", self.msh.ufl_cell(), self.ospace.k, 2)
        mel_coupling = ufl.MixedElement([vel_coupling, vel_coupling])
        self.fes_coupling = fem.FunctionSpace(self.msh, mel_coupling)
        self.dofmaps_coupling = [
            [self.fes_coupling.sub(j).sub(i).collapse()[1] for i in range(2)]
            for j in range(2)
        ]

        # For mass matrix between slabs
        vel_slab_bnd = ufl.VectorElement("CG", self.msh.ufl_cell(), self.ospace.k, 1)
        mel_slab_bnd = ufl.MixedElement([vel_slab_bnd, vel_slab_bnd])
        self.fes_slab_bnd = fem.FunctionSpace(self.msh, mel_slab_bnd)
        self.dofmaps_slab_bnd = [
            self.fes_slab_bnd.sub(i).collapse()[1] for i in range(2)
        ]
        self.u_0_slab_bnd = fem.Function(self.fes_slab_bnd)

        # For slab
        vel_pre_q = ufl.VectorElement(
            "CG",
            self.msh.ufl_cell(),
            self.ospace.k,
            self.otime.q + 1,
        )
        vel_pre_qstar = ufl.VectorElement(
            "CG",
            self.msh.ufl_cell(),
            self.ospace.kstar,
            self.otime.qstar + 1,
        )
        mel_pre = ufl.MixedElement([vel_pre_q, vel_pre_q, vel_pre_qstar, vel_pre_qstar])
        self.fes_p = fem.FunctionSpace(self.msh, mel_pre)
        self.dofmaps_pre = [
            [
                self.fes_p.sub(j).sub(i).collapse()[1]
                for i in range(self.subspace_time_order[j] + 1)
            ]
            for j in range(4)
        ]

        self.w1_p, self.w2_p, self.y1_p, self.y2_p = ufl.TestFunctions(self.fes_p)
        self.fes_u1_0_pre, self.dofmap_fes_u1_0_pre = (
            self.fes_p.sub(0).sub(self.otime.q).collapse()
        )  # for top of slice
        self.fes_u2_0_pre, self.dofmap_fes_u2_0_pre = (
            self.fes_p.sub(1).sub(self.otime.q).collapse()
        )  # for top of slice
        self.uh_pre = fem.Function(self.fes_p)
        self.u1_0_pre = fem.Function(self.fes_u1_0_pre)
        self.u2_0_pre = fem.Function(self.fes_u2_0_pre)

    def setup_spacetime_matrix(  # noqa: C901, PLR0912, PLR0915
        self,
        afes,
        precond=False,
    ):
        # TODO: this single function is far too long with too much complexity
        # (score 20!) and should be split up into subfunctions. Also too many
        # conditionals.
        #
        #  - C901 function is too complex
        #  - PLR0912 there are too many branches
        #  - PLR0915 there are too many conditional statements
        u1, u2, z1, z2 = ufl.TrialFunctions(afes)
        w1, w2, y1, y2 = ufl.TestFunctions(afes)

        dx = ufl.Measure("dx", domain=afes.mesh, metadata=self.metadata)
        n_facet = ufl.FacetNormal(afes.mesh)
        h = ufl.CellDiameter(afes.mesh)

        delta_t = self.delta_t
        elmat_time = self.elmat_time

        if afes == self.fes:
            omega_ind = self.omega.indicator_function
        else:
            q_ind = fem.FunctionSpace(afes.mesh, ("DG", 0))
            omega_ind = fem.Function(q_ind)
            omega_ind.interpolate(self.omega.indicator_function)

        # retrieve stabilization parameter
        gamma_data = self.stabilisation_terms["data"]
        gamma_dual = self.stabilisation_terms["dual"]
        gamma_primal = self.stabilisation_terms["primal"]
        gamma_primal_jump = self.stabilisation_terms["primal-jump"]

        int_idx_q = 0
        coupling_idx_q = -1
        int_idx_qstar = 0
        coupling_idx_qstar = -1

        a = 1e-20 * u1[0] * y1[0] * dx

        for n in range(self.N):
            n * delta_t

            for j in range(self.otime.q + 1):
                for k in range(self.otime.qstar + 1):
                    a += (
                        elmat_time["DM_time_q_qstar"][k, j]
                        * inner(u2[int_idx_q + j], y1[int_idx_qstar + k])
                        * dx
                    )
                    a += (
                        delta_t
                        * self.elmat_time["M_time_q_qstar"][k, j]
                        * inner(grad(u1[int_idx_q + j]), grad(y1[int_idx_qstar + k]))
                        * dx
                    )
                    a += (
                        elmat_time["DM_time_q_qstar"][k, j]
                        * inner(u1[int_idx_q + j], y2[int_idx_qstar + k])
                        * dx
                    )
                    a -= (
                        delta_t
                        * elmat_time["M_time_q_qstar"][k, j]
                        * inner(u2[int_idx_q + j], y2[int_idx_qstar + k])
                        * dx
                    )
                    if abs(elmat_time["M_time_q_qstar"][k, j]) > 1e-10:
                        a -= (
                            delta_t
                            * elmat_time["M_time_q_qstar"][k, j]
                            * inner(n_facet, grad(u1[int_idx_q + j]))
                            * y1[int_idx_qstar + k]
                            * ds
                        )  # Nitsche term

            for j in range(self.otime.q + 1):
                for k in range(self.otime.q + 1):
                    a += (
                        gamma_data
                        * delta_t
                        * elmat_time["M_time_q_q"][k, j]
                        * omega_ind
                        * inner(u1[int_idx_q + j], w1[int_idx_q + k])
                        * dx
                    )

            for j in range(self.otime.q + 1):
                for k in range(self.otime.qstar + 1):
                    a += (
                        elmat_time["DM_time_q_qstar"][k, j]
                        * inner(w2[int_idx_q + j], z1[int_idx_qstar + k])
                        * dx
                    )
                    a += (
                        delta_t
                        * elmat_time["M_time_q_qstar"][k, j]
                        * inner(grad(w1[int_idx_q + j]), grad(z1[int_idx_qstar + k]))
                        * dx
                    )
                    a += (
                        elmat_time["DM_time_q_qstar"][k, j]
                        * inner(w1[int_idx_q + j], z2[int_idx_qstar + k])
                        * dx
                    )
                    a -= (
                        delta_t
                        * elmat_time["M_time_q_qstar"][k, j]
                        * inner(w2[int_idx_q + j], z2[int_idx_qstar + k])
                        * dx
                    )
                    if abs(elmat_time["M_time_q_qstar"][k, j]) > 1e-10:
                        a -= (
                            delta_t
                            * elmat_time["M_time_q_qstar"][k, j]
                            * inner(n_facet, grad(w1[int_idx_q + j]))
                            * z1[int_idx_qstar + k]
                            * ds
                        )  # Nitsche term

            for j in range(self.otime.qstar + 1):
                for k in range(self.otime.qstar + 1):
                    a -= (
                        gamma_dual
                        * delta_t
                        * elmat_time["M_time_qstar_qstar"][k, j]
                        * inner(y1[int_idx_qstar + j], z1[int_idx_qstar + k])
                        * dx
                    )
                    a -= (
                        gamma_dual
                        * delta_t
                        * elmat_time["M_time_qstar_qstar"][k, j]
                        * inner(
                            grad(y1[int_idx_qstar + j]),
                            grad(z1[int_idx_qstar + k]),
                        )
                        * dx
                    )
                    a -= (
                        gamma_dual
                        * delta_t
                        * elmat_time["M_time_qstar_qstar"][k, j]
                        * inner(y2[int_idx_qstar + j], z2[int_idx_qstar + k])
                        * dx
                    )
                    a -= (
                        gamma_dual
                        * delta_t
                        * (self.lam_Nitsche / h)
                        * elmat_time["M_time_qstar_qstar"][k, j]
                        * inner(y1[int_idx_qstar + j], z1[int_idx_qstar + k])
                        * ds
                    )
            if self.jumps_in_fw_problem and n > 0:
                a -= (
                    gamma_dual
                    * delta_t
                    * inner(y2[coupling_idx_qstar + 1], z2[coupling_idx_qstar + 1])
                    * dx
                )
                a -= (
                    gamma_dual
                    * delta_t
                    * inner(y1[coupling_idx_qstar + 1], z1[coupling_idx_qstar + 1])
                    * dx
                )

            for j in range(self.otime.q + 1):
                for k in range(self.otime.q + 1):
                    # boundary term
                    a += (
                        gamma_primal
                        * delta_t
                        * (self.lam_Nitsche / h)
                        * elmat_time["M_time_q_q"][k, j]
                        * inner(u1[int_idx_q + j], w1[int_idx_q + k])
                        * ds
                    )
                    a += (
                        gamma_primal
                        * delta_t
                        * elmat_time["M_time_q_q"][k, j]
                        * 0.5
                        * (h("+") + h("-"))
                        * inner(
                            jump(grad(u1[int_idx_q + j])),
                            jump(grad(w1[int_idx_q + k])),
                        )
                        * dS
                    )
                    a += (
                        gamma_primal
                        * delta_t
                        * elmat_time["M_time_q_q"][k, j]
                        * inner(u2[int_idx_q + j], w2[int_idx_q + k])
                        * dx
                    )
                    a -= (
                        gamma_primal
                        * (elmat_time["DM_time_q_q"].T)[k, j]
                        * inner(u2[int_idx_q + j], w1[int_idx_q + k])
                        * dx
                    )
                    a -= (
                        gamma_primal
                        * elmat_time["DM_time_q_q"][k, j]
                        * inner(u1[int_idx_q + j], w2[int_idx_q + k])
                        * dx
                    )
                    a += (
                        gamma_primal
                        * (1 / delta_t)
                        * elmat_time["DDM_time_q_q"][k, j]
                        * inner(u1[int_idx_q + j], w1[int_idx_q + k])
                        * dx
                    )
                    a += (
                        gamma_primal
                        * h**2
                        * (1 / delta_t)
                        * elmat_time["DDM_time_q_q"][k, j]
                        * inner(u2[int_idx_q + j], w2[int_idx_q + k])
                        * dx
                    )
                    a -= (
                        gamma_primal
                        * h**2
                        * elmat_time["DM_time_q_q"][k, j]
                        * inner(u2[int_idx_q + j], div(grad(w1[int_idx_q + k])))
                        * dx
                    )
                    a -= (
                        gamma_primal
                        * h**2
                        * (elmat_time["DM_time_q_q"].T)[k, j]
                        * inner(div(grad(u1[int_idx_q + j])), w2[int_idx_q + k])
                        * dx
                    )
                    a += (
                        gamma_primal
                        * h**2
                        * delta_t
                        * elmat_time["M_time_q_q"][k, j]
                        * inner(
                            div(grad(u1[int_idx_q + j])),
                            div(grad(w1[int_idx_q + k])),
                        )
                        * dx
                    )

            if n == 0 and self.well_posed:
                print("Adding initial data.")
                a += (
                    gamma_primal_jump * (1 / delta_t) * inner(u1[0], w1[0]) * dx
                    + gamma_primal_jump * delta_t * inner(grad(u1[0]), grad(w1[0])) * dx
                )
                a += gamma_primal_jump * (1 / delta_t) * inner(u2[0], w2[0]) * dx

            # coupling terms in forward problem
            if self.jumps_in_fw_problem and n > 0:
                a += (
                    (1.0 + self.eps)
                    * inner(
                        w2[coupling_idx_q + 1] - w2[coupling_idx_q],
                        z1[coupling_idx_qstar + 1],
                    )
                    * dx
                )
                a += (
                    (1.0 + self.eps)
                    * inner(
                        w1[coupling_idx_q + 1] - w1[coupling_idx_q],
                        z2[coupling_idx_qstar + 1],
                    )
                    * dx
                )
                a += (
                    (1.0 + self.eps)
                    * inner(
                        u1[coupling_idx_q + 1] - u1[coupling_idx_q],
                        y2[coupling_idx_qstar + 1],
                    )
                    * dx
                )
                a += (
                    (1.0 + self.eps)
                    * inner(
                        u2[coupling_idx_q + 1] - u2[coupling_idx_q],
                        y1[coupling_idx_qstar + 1],
                    )
                    * dx
                )

            if n > 0:
                if precond:
                    a += (
                        gamma_primal_jump
                        * (1 / delta_t)
                        * inner(
                            u1[coupling_idx_q + 1] - u1[coupling_idx_q],
                            w1[coupling_idx_q + 1],
                        )
                        * dx
                    )
                    a += (
                        gamma_primal_jump
                        * delta_t
                        * inner(
                            grad(u1[coupling_idx_q + 1]) - grad(u1[coupling_idx_q]),
                            grad(w1[coupling_idx_q + 1]),
                        )
                        * dx
                    )
                    a += (
                        gamma_primal_jump
                        * (1 / delta_t)
                        * inner(
                            u2[coupling_idx_q + 1] - u2[coupling_idx_q],
                            w2[coupling_idx_q + 1],
                        )
                        * dx
                    )
                else:
                    a += (
                        gamma_primal_jump
                        * (1 / delta_t)
                        * inner(
                            u1[coupling_idx_q + 1] - u1[coupling_idx_q],
                            w1[coupling_idx_q + 1] - w1[coupling_idx_q],
                        )
                        * dx
                    )
                    a += (
                        gamma_primal_jump
                        * delta_t
                        * inner(
                            grad(u1[coupling_idx_q + 1]) - grad(u1[coupling_idx_q]),
                            grad(w1[coupling_idx_q + 1]) - grad(w1[coupling_idx_q]),
                        )
                        * dx
                    )
                    a += (
                        gamma_primal_jump
                        * (1 / delta_t)
                        * inner(
                            u2[coupling_idx_q + 1] - u2[coupling_idx_q],
                            w2[coupling_idx_q + 1] - w2[coupling_idx_q],
                        )
                        * dx
                    )

            int_idx_q += self.otime.q + 1
            coupling_idx_q += self.otime.q + 1
            int_idx_qstar += self.otime.qstar + 1
            coupling_idx_qstar += self.otime.qstar + 1

        print("Creating bfi")
        bilinear_form = fem.form(a)
        print("assemble matrix")
        A = assemble_matrix(bilinear_form, bcs=[])  # noqa: N806 | convention Ax = b
        A.assemble()

        return A, bilinear_form

    def get_spacetime_matrix(self, precond=False):
        if not self.spacetime_matrix:
            A, bilinear_form = self.setup_spacetime_matrix(  # noqa: N806
                self.fes,
                precond,
            )
            # TODO: was this a bug? also on l663 👇
            self.spacetime_bilinear_form = bilinear_form
            self.spacetime_matrix = A
        return self.spacetime_matrix

    def get_spacetime_bfi(self, precond=False):
        if not self.spacetime_bilinear_form:
            A, bilinear_form = self.setup_spacetime_matrix(  # noqa: N806
                self.fes,
                precond,
            )
            self.spacetime_bilinear_form = bilinear_form
            self.spacetime_matrix = A
        return self.spacetime_bilinear_form

    def setup_rhs(self):
        w1, w2, y1, y2 = ufl.TestFunctions(self.fes)

        dx = self.dx
        delta_t = self.delta_t
        gamma_primal_jump = self.stabilisation_terms["primal-jump"]
        gamma_data = self.stabilisation_terms["data"]

        int_idx_q = 0
        coupling_idx_q = -1
        int_idx_qstar = 0
        coupling_idx_qstar = -1

        L = 1e-20 * inner(1.0, y1[0]) * dx  # noqa: N806 | convention LU

        for n in range(self.N):
            t_n = n * delta_t

            # right hand side
            for tau_i, omega_i in zip(self.qr.current_pts(0, 1), self.qr.t_weights(1)):
                time_ti = t_n + delta_t * tau_i
                sol_ti = self.solution.v(time_ti, self.x)
                L += (  # noqa: N806 | convention LU
                    gamma_data
                    * delta_t
                    * omega_i
                    * sol_ti
                    * self.omega.indicator_function
                    * sum(
                        [
                            w1[int_idx_q + k] * self.phi_test[k](tau_i)
                            for k in range(self.otime.q + 1)
                        ],
                    )
                    * dx
                )

            if n == 0 and self.well_posed:
                print("Adding initial data on right hand side")
                L += (  # noqa: N806 | convention LU
                    gamma_primal_jump * (1 / delta_t) * inner(self.u1_0, w1[0]) * dx
                    + gamma_primal_jump
                    * delta_t
                    * inner(grad(self.u1_0), grad(w1[0]))
                    * dx
                )
                L += (  # noqa: N806 | convention LU
                    gamma_primal_jump * (1 / delta_t) * inner(self.u2_0, w2[0]) * dx
                )
                # TODO: check with Janosch, was this a bug? u1_0 and u2_0 were undefined.

            int_idx_q += self.otime.q + 1
            coupling_idx_q += self.otime.q + 1
            int_idx_qstar += self.otime.qstar + 1
            coupling_idx_qstar += self.otime.qstar + 1

        linear_form = fem.form(L)
        b_rhs = fem.petsc.create_vector(linear_form)
        fem.petsc.create_vector(linear_form)
        fem.petsc.assemble_vector(b_rhs, linear_form)

        self.SpaceTimeLfi = linear_form
        self.SpaceTimeRhs = b_rhs

    def get_spacetime_rhs(self):
        if not self.SpaceTimeRhs:
            self.setup_rhs()
        return self.SpaceTimeRhs

    def get_spacetime_lfi(self):
        if not self.SpaceTimeLfi:
            self.setup_rhs()
        return self.SpaceTimeLfi

    def setup_coupling_matrix_between_slices(self):
        u1_c, u2_c = ufl.TrialFunctions(self.fes_coupling)
        w1_c, w2_c = ufl.TestFunctions(self.fes_coupling)

        dx = self.dx
        delta_t = self.delta_t
        gamma_primal_jump = self.stabilisation_terms["primal-jump"]

        a_coupling = 1e-20 * u1_c[0] * w1_c[0] * dx
        a_coupling += (
            gamma_primal_jump
            * (1 / delta_t)
            * inner(u1_c[1] - u1_c[0], w1_c[1] - w1_c[0])
            * dx
        )
        a_coupling += (
            gamma_primal_jump
            * delta_t
            * inner(grad(u1_c[1]) - grad(u1_c[0]), grad(w1_c[1]) - grad(w1_c[0]))
            * dx
        )
        a_coupling += (
            gamma_primal_jump
            * (1 / delta_t)
            * inner(u2_c[1] - u2_c[0], w2_c[1] - w2_c[0])
            * dx
        )

        bfi_coupling = fem.form(a_coupling)
        A_coupling = assemble_matrix(bfi_coupling, bcs=[])  # noqa: N806
        A_coupling.assemble()

        self.coupling_matrix_between_slices = A_coupling
        (
            self.vec_coupling1,
            self.vec_coupling2,
        ) = self.coupling_matrix_between_slices.createVecs()

    def setup_scaled_mass_matrix_between_slices(self):
        u0_bnd = ufl.TrialFunctions(self.fes_slab_bnd)
        v0_bnd = ufl.TestFunctions(self.fes_slab_bnd)

        dx = self.dx
        delta_t = self.delta_t
        gamma_primal_jump = self.stabilisation_terms["primal-jump"]

        m_bnd = gamma_primal_jump * (1 / delta_t) * inner(u0_bnd[0], v0_bnd[0]) * dx
        m_bnd += (
            gamma_primal_jump * delta_t * inner(grad(u0_bnd[0]), grad(v0_bnd[0])) * dx
        )
        m_bnd += gamma_primal_jump * (1 / delta_t) * inner(u0_bnd[1], v0_bnd[1]) * dx

        bfi_m = fem.form(m_bnd)
        M_bnd = assemble_matrix(bfi_m, bcs=[])  # noqa: N806
        M_bnd.assemble()

        self.scaled_mass_matrix_between_slices = M_bnd
        (
            self.vec_0_bnd_in,
            self.vec_0_bnd_out,
        ) = self.scaled_mass_matrix_between_slices.createVecs()

    def setup_slab_matrix(self, with_jumps=True):  # noqa: C901, PLR0912, PLR0915
        # TODO: this function is too complex (complexity 19) and has too many
        # branches and conditional statements. Try to reduce the complexity on
        # refactoring and remove the suppressions.
        #
        #  - C901 function is too complex
        #  - PLR0912 there are too many branches
        #  - PLR0915 there are too many conditional statements
        u1_p, u2_p, z1_p, z2_p = ufl.TrialFunctions(self.fes_p)
        w1_p, w2_p, y1_p, y2_p = ufl.TestFunctions(self.fes_p)

        dx = self.dx
        n_facet = self.n_facet
        h = self.h
        delta_t = self.delta_t
        elmat_time = self.elmat_time

        # retrieve stabilization parameter
        gamma_data = self.stabilisation_terms["data"]
        gamma_dual = self.stabilisation_terms["dual"]
        gamma_primal = self.stabilisation_terms["primal"]
        gamma_primal_jump = self.stabilisation_terms["primal-jump"]

        a_pre = 1e-20 * u1_p[0] * y1_p[0] * dx

        for j in range(self.otime.q + 1):
            for k in range(self.otime.qstar + 1):
                a_pre += (
                    elmat_time["DM_time_q_qstar"][k, j] * inner(u2_p[j], y1_p[k]) * dx
                )
                a_pre += (
                    delta_t
                    * elmat_time["M_time_q_qstar"][k, j]
                    * inner(grad(u1_p[j]), grad(y1_p[k]))
                    * dx
                )
                a_pre += (
                    elmat_time["DM_time_q_qstar"][k, j] * inner(u1_p[j], y2_p[k]) * dx
                )
                a_pre -= (
                    delta_t
                    * elmat_time["M_time_q_qstar"][k, j]
                    * inner(u2_p[j], y2_p[k])
                    * dx
                )
                if abs(elmat_time["M_time_q_qstar"][k, j]) > 1e-10:
                    a_pre -= (
                        delta_t
                        * elmat_time["M_time_q_qstar"][k, j]
                        * inner(n_facet, grad(u1_p[j]))
                        * y1_p[k]
                        * ds
                    )  # Nitsche term

        for j in range(self.otime.q + 1):
            for k in range(self.otime.q + 1):
                a_pre += (
                    gamma_data
                    * delta_t
                    * elmat_time["M_time_q_q"][k, j]
                    * self.omega.indicator_function
                    * inner(u1_p[j], w1_p[k])
                    * dx
                )

        for j in range(self.otime.q + 1):
            for k in range(self.otime.qstar + 1):
                a_pre += (
                    elmat_time["DM_time_q_qstar"][k, j] * inner(w2_p[j], z1_p[k]) * dx
                )
                a_pre += (
                    delta_t
                    * elmat_time["M_time_q_qstar"][k, j]
                    * inner(grad(w1_p[j]), grad(z1_p[k]))
                    * dx
                )
                a_pre += (
                    elmat_time["DM_time_q_qstar"][k, j] * inner(w1_p[j], z2_p[k]) * dx
                )
                a_pre -= (
                    delta_t
                    * elmat_time["M_time_q_qstar"][k, j]
                    * inner(w2_p[j], z2_p[k])
                    * dx
                )
                if abs(elmat_time["M_time_q_qstar"][k, j]) > 1e-10:
                    a_pre -= (
                        delta_t
                        * elmat_time["M_time_q_qstar"][k, j]
                        * inner(n_facet, grad(w1_p[j]))
                        * z1_p[k]
                        * ds
                    )  # Nitsche term

        for j in range(self.otime.qstar + 1):
            for k in range(self.otime.qstar + 1):
                a_pre -= (
                    gamma_dual
                    * delta_t
                    * elmat_time["M_time_qstar_qstar"][k, j]
                    * inner(y1_p[j], z1_p[k])
                    * dx
                )
                a_pre -= (
                    gamma_dual
                    * delta_t
                    * elmat_time["M_time_qstar_qstar"][k, j]
                    * inner(grad(y1_p[j]), grad(z1_p[k]))
                    * dx
                )
                a_pre -= (
                    gamma_dual
                    * delta_t
                    * elmat_time["M_time_qstar_qstar"][k, j]
                    * inner(y2_p[j], z2_p[k])
                    * dx
                )
                a_pre -= (
                    gamma_dual
                    * delta_t
                    * (self.lam_Nitsche / h)
                    * elmat_time["M_time_qstar_qstar"][k, j]
                    * inner(y1_p[j], z1_p[k])
                    * ds
                )

        if self.jumps_in_fw_problem and with_jumps:
            a_pre -= gamma_dual * delta_t * inner(y2_p[0], z2_p[0]) * dx
            a_pre -= gamma_dual * delta_t * inner(y1_p[0], z1_p[0]) * dx

        for j in range(self.otime.q + 1):
            for k in range(self.otime.q + 1):
                # boundary term
                a_pre += (
                    gamma_primal
                    * delta_t
                    * (self.lam_Nitsche / h)
                    * elmat_time["M_time_q_q"][k, j]
                    * inner(u1_p[j], w1_p[k])
                    * ds
                )
                a_pre += (
                    gamma_primal
                    * delta_t
                    * elmat_time["M_time_q_q"][k, j]
                    * 0.5
                    * (h("+") + h("-"))
                    * inner(jump(grad(u1_p[j])), jump(grad(w1_p[k])))
                    * dS
                )
                a_pre += (
                    gamma_primal
                    * delta_t
                    * elmat_time["M_time_q_q"][k, j]
                    * inner(u2_p[j], w2_p[k])
                    * dx
                )
                a_pre -= (
                    gamma_primal
                    * (elmat_time["DM_time_q_q"].T)[k, j]
                    * inner(u2_p[j], w1_p[k])
                    * dx
                )
                a_pre -= (
                    gamma_primal
                    * elmat_time["DM_time_q_q"][k, j]
                    * inner(u1_p[j], w2_p[k])
                    * dx
                )
                a_pre += (
                    gamma_primal
                    * (1 / delta_t)
                    * elmat_time["DDM_time_q_q"][k, j]
                    * inner(u1_p[j], w1_p[k])
                    * dx
                )
                a_pre += (
                    gamma_primal
                    * h**2
                    * (1 / delta_t)
                    * elmat_time["DDM_time_q_q"][k, j]
                    * inner(u2_p[j], w2_p[k])
                    * dx
                )
                a_pre -= (
                    gamma_primal
                    * h**2
                    * elmat_time["DM_time_q_q"][k, j]
                    * inner(u2_p[j], div(grad(w1_p[k])))
                    * dx
                )
                a_pre -= (
                    gamma_primal
                    * h**2
                    * (elmat_time["DM_time_q_q"].T)[k, j]
                    * inner(div(grad(u1_p[j])), w2_p[k])
                    * dx
                )
                a_pre += (
                    gamma_primal
                    * h**2
                    * delta_t
                    * elmat_time["M_time_q_q"][k, j]
                    * inner(div(grad(u1_p[j])), div(grad(w1_p[k])))
                    * dx
                )

        if with_jumps:
            a_pre += (
                gamma_primal_jump * (1 / delta_t) * inner(u1_p[0], w1_p[0]) * dx
                + gamma_primal_jump * delta_t * inner(grad(u1_p[0]), grad(w1_p[0])) * dx
            )
            a_pre += gamma_primal_jump * (1 / delta_t) * inner(u2_p[0], w2_p[0]) * dx

            if self.jumps_in_fw_problem:
                a_pre += (1.0 + self.eps) * inner(y1_p[0], u2_p[0]) * dx
                a_pre += (1.0 + self.eps) * inner(y2_p[0], u1_p[0]) * dx

        bfi_pre = fem.form(a_pre)
        A_pre = assemble_matrix(bfi_pre, bcs=[])  # noqa: N806
        A_pre.assemble()

        if with_jumps:
            self.slab_bilinear_form = bfi_pre
            self.slab_matrix = A_pre
            if self.well_posed:
                self.SlabBfiFirstSlab = bfi_pre
                self.SlabMatFirstSlab = A_pre
        else:
            self.slab_bilinear_form_no_dg_jumps = bfi_pre
            self.slab_matrix_no_dg_jumps = A_pre
            self.vec_slab1, self.vec_slab2 = A_pre.createVecs()
            if not self.well_posed:
                self.SlabBfiFirstSlab = bfi_pre
                self.SlabMatFirstSlab = A_pre

    def get_slab_bfi(self):
        if not self.slab_bilinear_form:
            self.setup_slab_matrix(with_jumps=True)
            self.setup_slab_matrix(with_jumps=False)
        return self.slab_bilinear_form

    def get_slab_matrix(self):
        if not self.slab_matrix:
            self.setup_slab_matrix(with_jumps=True)
            self.setup_slab_matrix(with_jumps=False)
        return self.slab_matrix

    def get_slab_bfi_first_slab(self):
        return self.SlabBfiFirstSlab

    def get_slab_matrix_first_slab(self):
        return self.SlabMatFirstSlab

    def set_solver_first_slab(self, solver):
        # TODO: much of these getters and setters could be @property's
        self.SolverFirstSlab = solver

    def set_solver_slab(self, solver):
        self.solver_slab = solver

    def slab_matrix_no_dg_jumps_multiply(self, vec_in, vec_out):
        # TODO: check my understanding of this with Janosch - probably could do with a docstring
        # if self.mkl_matrix_mult:
        self.slab_matrix_no_dg_jumps.mult(vec_in, vec_out)

    def coupling_matrix_between_slices_multiply(self, vec_in, vec_out):
        self.coupling_matrix_between_slices.mult(vec_in, vec_out)

    def apply_spacetime_matrix(self, vec_in, vec_out):
        vec_out.array[:] = 0.0

        int_idx_q = 0
        coupling_idx_q = -1
        int_idx_qstar = 0
        coupling_idx_qstar = -1

        for n in range(self.N):
            for j in range(4):
                shift_idx = int_idx_q if j in [0, 1] else int_idx_qstar
                for ell in range(self.subspace_time_order[j] + 1):
                    self.vec_slab1.array[self.dofmaps_pre[j][ell]] = vec_in.array[
                        self.dofmaps_full[j][shift_idx + ell]
                    ]

            self.slab_matrix_no_dg_jumps_multiply(
                self.vec_slab1,
                self.vec_slab2,
            )  # multiplication with slab matrix

            for j in range(4):
                shift_idx = int_idx_q if j in [0, 1] else int_idx_qstar
                for ell in range(self.subspace_time_order[j] + 1):
                    vec_out.array[
                        self.dofmaps_full[j][shift_idx + ell]
                    ] += self.vec_slab2.array[self.dofmaps_pre[j][ell]]

            if n > 0:
                for j in range(2):
                    for i in range(2):
                        self.vec_coupling1.array[
                            self.dofmaps_coupling[j][i]
                        ] = vec_in.array[self.dofmaps_full[j][coupling_idx_q + i]]

                self.coupling_matrix_between_slices_multiply(
                    self.vec_coupling1,
                    self.vec_coupling2,
                )

                for j in range(2):
                    for i in range(2):
                        vec_out.array[
                            self.dofmaps_full[j][coupling_idx_q + i]
                        ] += self.vec_coupling2.array[self.dofmaps_coupling[j][i]]

            int_idx_q += self.otime.q + 1
            coupling_idx_q += self.otime.q + 1
            int_idx_qstar += self.otime.qstar + 1
            coupling_idx_qstar += self.otime.qstar + 1

    def pre_time_marching(self, b, x_sol):
        dx = self.dx
        delta_t = self.delta_t

        int_idx_q = 0
        coupling_idx_q = -1
        int_idx_qstar = 0
        coupling_idx_qstar = -1

        # retrieve stabilization parameter
        gamma_primal_jump = self.stabilisation_terms["primal-jump"]

        for n in range(self.N):
            n * delta_t

            l_pre = 1e-20 * inner(1.0, self.y1_p[0]) * dx

            if n > 0:
                l_pre += (
                    gamma_primal_jump
                    * (1 / delta_t)
                    * inner(self.u1_0_pre, self.w1_p[0])
                    * dx
                    + gamma_primal_jump
                    * delta_t
                    * inner(grad(self.u1_0_pre), grad(self.w1_p[0]))
                    * dx
                )
                l_pre += (
                    gamma_primal_jump
                    * (1 / delta_t)
                    * inner(self.u2_0_pre, self.w2_p[0])
                    * dx
                )

                if self.jumps_in_fw_problem:
                    l_pre += inner(self.u2_0_pre, self.y1_p[0]) * dx
                    l_pre += inner(self.u1_0_pre, self.y2_p[0]) * dx

            lfi_pre = fem.form(l_pre)
            b_rhs_pre = fem.petsc.create_vector(lfi_pre)
            fem.petsc.assemble_vector(b_rhs_pre, lfi_pre)

            # copy rhs entries from global to local vector on the slice
            for j in range(4):
                shift_idx = int_idx_q if j in [0, 1] else int_idx_qstar
                for ell in range(self.subspace_time_order[j] + 1):
                    b_rhs_pre.array[self.dofmaps_pre[j][ell]] += b.array[
                        self.dofmaps_full[j][shift_idx + ell]
                    ]

            if n == 0:
                self.SolverFirstSlab.solve(b_rhs_pre, self.uh_pre.vector)
                self.uh_pre.x.scatter_forward()
            else:
                self.solver_slab.solve(b_rhs_pre, self.uh_pre.vector)
                self.uh_pre.x.scatter_forward()

            u1_h_p, u2_h_p, z1_h_p, z2_h_p = self.uh_pre.split()
            self.u1_0_pre.x.array[:] = u1_h_p.x.array[self.dofmaps_pre[0][self.otime.q]]
            self.u2_0_pre.x.array[:] = u2_h_p.x.array[self.dofmaps_pre[1][self.otime.q]]

            for j in range(4):
                shift_idx = int_idx_q if j in [0, 1] else int_idx_qstar
                for ell in range(self.subspace_time_order[j] + 1):
                    x_sol.array[
                        self.dofmaps_full[j][shift_idx + ell]
                    ] = self.uh_pre.x.array[self.dofmaps_pre[j][ell]]

            int_idx_q += self.otime.q + 1
            coupling_idx_q += self.otime.q + 1
            int_idx_qstar += self.otime.qstar + 1
            coupling_idx_qstar += self.otime.qstar + 1

    def pre_time_marching_improved(self, b, x_sol):
        if self.jumps_in_fw_problem:
            msg = "pre_time_marching_improved not implemented for jumps_in_fw_problem==True"
            raise ValueError(msg)

        delta_t = self.delta_t

        int_idx_q = 0
        coupling_idx_q = -1
        int_idx_qstar = 0
        coupling_idx_qstar = -1

        # retrieve stabilization parameter
        self.stabilisation_terms["primal-jump"]  # TODO: check why this is here

        self.b_rhs_pre.array[:] = 0.0

        for n in range(self.N):
            n * delta_t

            # copy rhs entries from global to local vector on the slice
            for j in range(4):
                shift_idx = int_idx_q if j in [0, 1] else int_idx_qstar
                for ell in range(self.subspace_time_order[j] + 1):
                    self.b_rhs_pre.array[self.dofmaps_pre[j][ell]] += b.array[
                        self.dofmaps_full[j][shift_idx + ell]
                    ]

            if n == 0:
                self.SolverFirstSlab.solve(self.b_rhs_pre, self.uh_pre.vector)
                self.uh_pre.x.scatter_forward()
            else:
                self.solver_slab.solve(self.b_rhs_pre, self.uh_pre.vector)
                self.uh_pre.x.scatter_forward()

            u1_h_p, u2_h_p, z1_h_p, z2_h_p = self.uh_pre.split()

            self.vec_0_bnd_in.array[:] = 0.0
            self.vec_0_bnd_in.array[self.dofmaps_slab_bnd[0]] += u1_h_p.x.array[
                self.dofmaps_pre[0][self.otime.q]
            ]
            self.vec_0_bnd_in.array[self.dofmaps_slab_bnd[1]] += u2_h_p.x.array[
                self.dofmaps_pre[1][self.otime.q]
            ]
            self.scaled_mass_matrix_between_slices.mult(
                self.vec_0_bnd_in,
                self.vec_0_bnd_out,
            )

            self.b_rhs_pre.array[:] = 0.0
            self.b_rhs_pre.array[self.dofmaps_pre[0][0]] += self.vec_0_bnd_out.array[
                self.dofmaps_slab_bnd[0]
            ]
            self.b_rhs_pre.array[self.dofmaps_pre[1][0]] += self.vec_0_bnd_out.array[
                self.dofmaps_slab_bnd[1]
            ]

            for j in range(4):
                shift_idx = int_idx_q if j in [0, 1] else int_idx_qstar
                for ell in range(self.subspace_time_order[j] + 1):
                    x_sol.array[
                        self.dofmaps_full[j][shift_idx + ell]
                    ] = self.uh_pre.x.array[self.dofmaps_pre[j][ell]]

            int_idx_q += self.otime.q + 1
            coupling_idx_q += self.otime.q + 1
            int_idx_qstar += self.otime.qstar + 1
            coupling_idx_qstar += self.otime.qstar + 1

    def get_spacetime_matrix_as_linear_operator(self):
        return SpaceTime.FMatrix(self)

    # TODO: why do we need this class? and why is it nested?
    class FMatrix:
        def __init__(self, st_instance):
            self.st = st_instance

        def mult(self, vec_in, vec_out):
            self.st.apply_spacetime_matrix(vec_in, vec_out)

        def create_vectors(self):
            tmp1 = fem.petsc.create_vector(self.st.SpaceTimeLfi)
            tmp2 = fem.petsc.create_vector(self.st.SpaceTimeLfi)
            return tmp1, tmp2

    def prepare_precondition_gmres(self):
        if not self.slab_matrix:
            self.setup_slab_matrix(with_jumps=True)
            self.b_rhs_pre, _ = self.slab_matrix.createVecs()
        if not self.SlabMatFirstSlab and not self.well_posed:
            self.setup_slab_matrix(with_jumps=False)
        if not self.coupling_matrix_between_slices:
            self.setup_coupling_matrix_between_slices()
        if not self.SpaceTimeRhs:
            self.setup_rhs()
        if not self.scaled_mass_matrix_between_slices:
            self.setup_scaled_mass_matrix_between_slices()
        # if self.mkl_matrix_mult:

    def prepare_coarse_grid_correction(self, msh_coarse):
        self.msh_coarse = msh_coarse
        vel_coarse_q = ufl.VectorElement(
            "CG",
            self.msh_coarse.ufl_cell(),
            self.ospace.k,
            int((self.otime.q + 1) * self.N),
        )
        vel_coarse_qstar = ufl.VectorElement(
            "CG",
            self.msh_coarse.ufl_cell(),
            self.ospace.kstar,
            int((self.otime.qstar + 1) * self.N),
        )
        mel_coarse = ufl.MixedElement(
            [vel_coarse_q, vel_coarse_q, vel_coarse_qstar, vel_coarse_qstar],
        )
        self.fes_coarse = fem.FunctionSpace(self.msh_coarse, mel_coarse)

        dofmaps_full_coarse = []
        for j in range(4):
            dofmaps_j_coarse = []
            int_idx_q = 0
            int_idx_qstar = 0
            for _n in range(self.N):
                for ell in range(self.subspace_time_order[j] + 1):
                    shift_idx = int_idx_q if j in [0, 1] else int_idx_qstar
                    dofmaps_j_coarse.append(
                        self.fes_coarse.sub(j).sub(shift_idx + ell).collapse()[1],
                    )
                int_idx_q += self.otime.q + 1
                int_idx_qstar += self.otime.qstar + 1
            dofmaps_full_coarse.append(dofmaps_j_coarse)
        self.dofmaps_full_coarse = dofmaps_full_coarse

        self.fes_node_coarse, self.dofmap_node_coarse = (
            self.fes_coarse.sub(0).sub(0).collapse()
        )
        self.fes_node_fine, self.dofmap_node_fine = self.fes.sub(0).sub(0).collapse()
        self.u_node_fine = fem.Function(self.fes_node_fine)
        self.u_node_coarse = fem.Function(self.fes_node_coarse)

        self.SpaceTimeMatCoarse, self.SpaceTimeBfiCoarse = self.setup_spacetime_matrix(
            self.fes_coarse,
            precond=False,
        )

        self.tmp1_fine = fem.petsc.create_vector(self.SpaceTimeLfi)
        self.tmp2_fine = fem.petsc.create_vector(self.SpaceTimeLfi)
        self.tmp3_sol = fem.petsc.create_vector(self.SpaceTimeLfi)
        self.tmp4_sol = fem.petsc.create_vector(self.SpaceTimeLfi)
        self.tmp5_sol = fem.petsc.create_vector(self.SpaceTimeLfi)
        self.tmp6_sol = fem.petsc.create_vector(self.SpaceTimeLfi)
        self.tmp1_coarse, self.tmp2_coarse = self.SpaceTimeMatCoarse.createVecs()

    def get_spacetime_matrix_coarse(self):
        return self.SpaceTimeMatCoarse

    def set_solver_coarse(self, solver):
        self.SolverCoarse = solver

    def prolongation(self, x_coarse, x_fine):
        int_idx_q = 0
        int_idx_qstar = 0
        for _n in range(self.N):
            for j in range(4):
                shift_idx = int_idx_q if j in [0, 1] else int_idx_qstar
                for ell in range(self.subspace_time_order[j] + 1):
                    self.u_node_coarse.x.array[:] = x_coarse.array[
                        self.dofmaps_full_coarse[j][shift_idx + ell]
                    ]
                    self.u_node_fine.interpolate(self.u_node_coarse)
                    x_fine.array[
                        self.dofmaps_full[j][shift_idx + ell]
                    ] = self.u_node_fine.x.array[:]
            int_idx_q += self.otime.q + 1
            int_idx_qstar += self.otime.qstar + 1

    def restriction(self, x_coarse, x_fine):
        int_idx_q = 0
        int_idx_qstar = 0
        for _n in range(self.N):
            for j in range(4):
                shift_idx = int_idx_q if j in [0, 1] else int_idx_qstar
                for ell in range(self.subspace_time_order[j] + 1):
                    self.u_node_fine.x.array[:] = x_fine.array[
                        self.dofmaps_full[j][shift_idx + ell]
                    ]
                    self.u_node_coarse.interpolate(self.u_node_fine)
                    x_coarse.array[
                        self.dofmaps_full_coarse[j][shift_idx + ell]
                    ] = self.u_node_coarse.x.array[:]
            int_idx_qstar += self.otime.qstar + 1
            int_idx_q += self.otime.q + 1

    def q_op(self, x_in, x_out):
        self.restriction(self.tmp1_coarse, x_in)
        self.SolverCoarse.solve(self.tmp1_coarse, self.tmp2_coarse)
        self.prolongation(self.tmp2_coarse, x_out)

    def pre_twolvl(self, b, x_sol):
        self.otime.q_op(b, self.tmp3_sol)
        self.apply_spacetime_matrix(self.tmp3_sol, self.tmp4_sol)
        self.tmp5_sol.array[:] = b.array[:]
        self.tmp5_sol.array[:] -= self.tmp4_sol.array[:]
        self.pre_time_marching(self.tmp5_sol, self.tmp6_sol)
        x_sol.array[:] = self.tmp6_sol.array[:]
        x_sol.array[:] += self.tmp3_sol.array[:]

    def measured_errors(self, u_inp, verbose=False):  # noqa: PLR0915
        # TODO: this function has too many statements, remove the suppression of
        # PLR0915 once refactored for simplicity.
        err_dict = {}
        dx = self.dx
        delta_t = self.delta_t

        self.uh.x.array[:] = u_inp.array[:]
        self.uh.x.scatter_forward()
        u1_h, u2_h, z1_h, z2_h = self.uh.split()

        self.u1_0.x.array[:] = u1_h.x.array[self.dofmaps_full[0][-1]]
        self.u2_0.x.array[:] = u2_h.x.array[self.dofmaps_full[1][-1]]
        ue = self.solution.v(self.T, self.x)
        l2_error = fem.form(ufl.inner(self.u1_0 - ue, self.u1_0 - ue) * dx)
        error_local = fem.assemble_scalar(l2_error)
        error_l2_abs = np.sqrt(self.msh.comm.allreduce(error_local, op=MPI.SUM))
        print(f"t = {self.T}, L2-error = {error_l2_abs} ")
        err_dict["L2-u1-T"] = error_l2_abs

        l2_error_at_samples = []
        l2_error_dt_at_samples = []

        int_idx = 0

        total_error_l2l2_dt = 0

        for n in range(self.N):
            t_n = n * delta_t
            if n > 0:
                self.u1_minus.x.array[:] = u1_h.x.array[
                    self.dofmaps_full[0][int_idx - 1]
                ]

            for ell in range(self.otime.q + 1):
                self.u1_slab[ell].x.array[:] = u1_h.x.array[
                    self.dofmaps_full[0][int_idx + ell]
                ]

            # measuring L2-error in time of u_t
            for tau_i, omega_i in zip(
                self.qr_ho.current_pts(0, 1),
                self.qr_ho.t_weights(1),
            ):
                time_ti = t_n + delta_t * tau_i
                dt_ue_tni = self.solution.dvdt(time_ti, self.x)
                uh_prime_at_ti = sum(
                    [
                        (1 / delta_t)
                        * self.dt_phi_trial[j]((time_ti - t_n) / delta_t)
                        * self.u1_slab[j]
                        for j in range(self.otime.q + 1)
                    ],
                )
                if n > 0:
                    uh_prime_at_ti -= (
                        (2 / delta_t)
                        * (self.u1_slab[0] - self.u1_minus)
                        * d_theta_ref((2 * time_ti - (t_n + delta_t + t_n)) / delta_t)
                    )

                l2_error_dt_tni = fem.form(
                    ufl.inner(uh_prime_at_ti - dt_ue_tni, uh_prime_at_ti - dt_ue_tni)
                    * dx,
                )
                error_dt_tni_local = fem.assemble_scalar(l2_error_dt_tni)
                error_dt_tni = self.msh.comm.allreduce(error_dt_tni_local, op=MPI.SUM)
                total_error_l2l2_dt += omega_i * delta_t * error_dt_tni

            for t_sample in self.sample_pts_error:
                if t_sample >= t_n and t_sample <= (t_n + delta_t):
                    ue = self.solution.v(t_sample, self.x)
                    dt_ue = self.solution.dvdt(t_sample, self.x)
                    uh_at_ti = sum(
                        [
                            self.phi_trial[j]((t_sample - t_n) / delta_t)
                            * self.u1_slab[j]
                            for j in range(self.otime.q + 1)
                        ],
                    )
                    uh_prime_at_ti = sum(
                        [
                            (1 / delta_t)
                            * self.dt_phi_trial[j]((t_sample - t_n) / delta_t)
                            * self.u1_slab[j]
                            for j in range(self.otime.q + 1)
                        ],
                    )
                    if n > 0:
                        uh_at_ti -= theta_ref(
                            (2 * t_sample - (t_n + delta_t + t_n)) / delta_t,
                        ) * (self.u1_slab[0] - self.u1_minus)
                        uh_prime_at_ti -= (
                            (2 / delta_t)
                            * (self.u1_slab[0] - self.u1_minus)
                            * d_theta_ref(
                                (2 * t_sample - (t_n + delta_t + t_n)) / delta_t,
                            )
                        )

                    l2_error = fem.form(ufl.inner(uh_at_ti - ue, uh_at_ti - ue) * dx)
                    l2_error_dt = fem.form(
                        ufl.inner(uh_prime_at_ti - dt_ue, uh_prime_at_ti - dt_ue) * dx,
                    )

                    error_local = fem.assemble_scalar(l2_error)
                    error_local_dt = fem.assemble_scalar(l2_error_dt)
                    error_l2_abs = np.sqrt(
                        self.msh.comm.allreduce(error_local, op=MPI.SUM),
                    )
                    error_dt_l2_abs = np.sqrt(
                        self.msh.comm.allreduce(error_local_dt, op=MPI.SUM),
                    )
                    l2_error_at_samples.append(error_l2_abs)
                    l2_error_dt_at_samples.append(error_dt_l2_abs)
                    if verbose:
                        print(
                            f"t = {t_sample}, L2-error u = {error_l2_abs}, L2-error u_t = {error_dt_l2_abs} ",
                        )

            int_idx += self.otime.q + 1

        print("L-infty-L2 error u = ", max(l2_error_at_samples))
        print("L-infty-L2 error u_t = ", max(l2_error_dt_at_samples))
        print("L2-L2 error u_t = ", sqrt(total_error_l2l2_dt))

        err_dict["L-infty-L2-error-u"] = max(l2_error_at_samples)
        err_dict["L-infty-L2-error-ut"] = max(l2_error_dt_at_samples)
        err_dict["L2-L2-error-u_t"] = sqrt(total_error_l2l2_dt)

        return err_dict

    def plot(  # noqa: PLR0915
        self,
        uh_inp,
        n_space=100,
        n_time_subdiv=10,
        abs_val=False,
    ):
        # TODO: this function has too many statements, remove the suppression of
        # PLR0915 once refactored for simplicity.
        trafo = (lambda x: abs(x)) if abs_val else lambda x: x

        uh_plot = fem.Function(self.fes)
        uh_plot.x.array[:] = uh_inp.array[:]
        u1_h, u2_h, z1_h, z2_h = uh_plot.split()

        u1_slab_plot = [fem.Function(self.fes_u1_0) for j in range(self.otime.q + 1)]

        delta_t = self.delta_t
        right_end = 1.0
        left_end = 0.0
        eval_pts_space = np.linspace(left_end, right_end, num=n_space).tolist()
        eval_pts_space_3d = np.array([(xp, 0, 0) for xp in eval_pts_space])

        eval_t_subdiv = np.linspace(
            0.0,
            1.0,
            num=n_time_subdiv,
            endpoint=False,
        ).tolist()

        ts = []

        int_idx = 0
        time_idx = 0

        fun_val = np.zeros((n_time_subdiv * self.N, n_space))
        dt_fun_val = np.zeros((n_time_subdiv * self.N, n_space))
        space_val_ti = np.zeros((self.otime.q + 1, n_space))

        for n in range(self.N):
            t_n = n * delta_t

            for ell in range(self.otime.q + 1):
                u1_slab_plot[ell].x.array[:] = u1_h.x.array[
                    self.dofmaps_full[0][int_idx + ell]
                ]

            for ell in range(self.otime.q + 1):
                space_val_ti[ell, :] = evaluate_function_at_points(
                    u1_slab_plot[ell],
                    eval_pts_space_3d,
                )

            # measuring L2-error in time of u_t
            for j in range(n_space):
                for m, tau_m in enumerate(eval_t_subdiv):
                    time_ti = t_n + delta_t * tau_m
                    if j == 0:
                        ts.append(time_ti)
                    uh_at_ti = trafo(
                        sum(
                            [
                                self.phi_trial[ell]((time_ti - t_n) / delta_t)
                                * space_val_ti[ell, j]
                                for ell in range(self.otime.q + 1)
                            ],
                        ),
                    )
                    dt_uh_at_ti = trafo(
                        sum(
                            [
                                (1 / delta_t)
                                * self.dt_phi_trial[ell]((time_ti - t_n) / delta_t)
                                * space_val_ti[ell, j]
                                for ell in range(self.otime.q + 1)
                            ],
                        ),
                    )
                    fun_val[time_idx + m, j] = uh_at_ti
                    dt_fun_val[time_idx + m, j] = dt_uh_at_ti

            int_idx += self.otime.q + 1
            time_idx += n_time_subdiv

        if False:
            vmin = np.min(fun_val) * 1e1
            vmax = np.max(fun_val) * 1e1

            im = plt.pcolormesh(
                eval_pts_space,
                ts,
                fun_val,
                cmap=plt.get_cmap("hot"),
                norm=colors.LogNorm(vmin=vmin, vmax=vmax),
            )
        else:
            cmap = plt.get_cmap("PiYG")
            levels = MaxNLocator(nbins=15).tick_values(
                1e4 * fun_val.min(),
                1e-1 * fun_val.max(),
            )
            norm = colors.BoundaryNorm(levels, ncolors=cmap.N, clip=True)
            im = plt.pcolormesh(eval_pts_space, ts, fun_val, cmap=cmap, norm=norm)
        plt.colorbar(im)
        plt.xlabel("x")
        plt.ylabel("t")
        plt.show()

        if False:
            vmin = np.min(fun_val) * 1e2
            vmax = np.max(fun_val) * 1e2
            im = plt.pcolormesh(
                eval_pts_space,
                ts,
                dt_fun_val,
                cmap=plt.get_cmap("hot"),
                norm=colors.LogNorm(vmin=vmin, vmax=vmax),
            )
        else:
            cmap = plt.get_cmap("jet")
            levels = MaxNLocator(nbins=15).tick_values(
                dt_fun_val.min(),
                dt_fun_val.max(),
            )
            norm = colors.BoundaryNorm(levels, ncolors=cmap.N, clip=True)
            im = plt.pcolormesh(eval_pts_space, ts, dt_fun_val, cmap=cmap, norm=norm)
        plt.colorbar(im)
        plt.xlabel("x")
        plt.ylabel("t")
        plt.show()

    def plot_error(self, uh_inp, n_space=100, n_time_subdiv=10):
        uh_plot = fem.Function(self.fes)
        uh_plot.x.array[:] = uh_inp.array[:]
        u1_h, u2_h, z1_h, z2_h = uh_plot.split()

        u1_slab_plot = [fem.Function(self.fes_u1_0) for j in range(self.otime.q + 1)]

        delta_t = self.delta_t
        right_end = 1.0
        left_end = 0.0
        eval_pts_space = np.linspace(left_end, right_end, num=n_space).tolist()
        eval_pts_space_3d = np.array([(xp, 0, 0) for xp in eval_pts_space])

        eval_t_subdiv = np.linspace(
            0.0,
            1.0,
            num=n_time_subdiv,
            endpoint=False,
        ).tolist()

        ts = []

        int_idx = 0
        time_idx = 0

        fun_val = np.zeros((n_time_subdiv * self.N, n_space))
        dt_fun_val = np.zeros((n_time_subdiv * self.N, n_space))
        err_val = np.zeros((n_time_subdiv * self.N, n_space))
        dt_err_val = np.zeros((n_time_subdiv * self.N, n_space))
        space_val_ti = np.zeros((self.otime.q + 1, n_space))

        for n in range(self.N):
            t_n = n * delta_t

            for i in range(self.otime.q + 1):
                u1_slab_plot[i].x.array[:] = u1_h.x.array[
                    self.dofmaps_full[0][int_idx + i]
                ]

            for i in range(self.otime.q + 1):
                space_val_ti[i, :] = evaluate_function_at_points(
                    u1_slab_plot[i],
                    eval_pts_space_3d,
                )

            # measuring L2-error in time of u_t
            for j in range(n_space):
                for m, tau_m in enumerate(eval_t_subdiv):
                    time_ti = t_n + delta_t * tau_m
                    if j == 0:
                        ts.append(time_ti)
                    uh_at_ti = sum(
                        [
                            self.phi_trial[ell]((time_ti - t_n) / delta_t)
                            * space_val_ti[ell, j]
                            for ell in range(self.otime.q + 1)
                        ],
                    )
                    u_at_ti = self.solution.v(time_ti, [eval_pts_space[j]])

                    dt_uh_at_ti = sum(
                        [
                            (1 / delta_t)
                            * self.dt_phi_trial[ell]((time_ti - t_n) / delta_t)
                            * space_val_ti[ell, j]
                            for ell in range(self.otime.q + 1)
                        ],
                    )
                    dt_u_at_ti = self.solution.dvdt(time_ti, [eval_pts_space[j]])

                    fun_val[time_idx + m, j] = uh_at_ti
                    err_val[time_idx + m, j] = abs(uh_at_ti - u_at_ti)

                    dt_fun_val[time_idx + m, j] = dt_uh_at_ti
                    dt_err_val[time_idx + m, j] = abs(dt_uh_at_ti - dt_u_at_ti)

            int_idx += self.otime.q + 1
            time_idx += n_time_subdiv

        plt.pcolormesh(eval_pts_space, ts, dt_fun_val)
        plt.show()

        vmin = 1e-4
        vmax = 1e-1
        im = plt.pcolormesh(
            eval_pts_space,
            ts,
            err_val,
            cmap=plt.get_cmap("hot"),
            norm=colors.LogNorm(vmin=vmin, vmax=vmax),
        )

        plt.colorbar(im)
        plt.xlabel("x")
        plt.ylabel("t")

        plt.show()

        vmin = 1e-4
        vmax = 1e-1

        im = plt.pcolormesh(
            eval_pts_space,
            ts,
            dt_err_val,
            cmap=plt.get_cmap("hot"),
            norm=colors.LogNorm(vmin=vmin, vmax=vmax),
        )
        plt.colorbar(im)
        plt.xlabel("x")
        plt.ylabel("t")

        plt.show()

    def plot_para_view(self, uh_inp, name="abserr"):
        self.uh.x.array[:] = uh_inp.array[:]
        self.uh.x.scatter_forward()
        u1_h, u2_h, z1_h, z2_h = self.uh.split()

        self.u1_0.x.array[:] = u1_h.x.array[self.dofmaps_full[0][-1]]
        self.u2_0.x.array[:] = u2_h.x.array[self.dofmaps_full[1][-1]]

        u_exact = fem.Function(self.fes_u1_0)
        u_diff = fem.Function(self.fes_u1_0)

        ut_exact = fem.Function(self.fes_u2_0)
        ut_diff = fem.Function(self.fes_u2_0)

        ue = self.solution.v(self.T, self.x)
        dt_ue = self.solution.dvdt(self.T, self.x)

        u_expr_exact = fem.Expression(ue, self.fes_u1_0.element.interpolation_points())
        u_exact.interpolate(u_expr_exact)

        ut_expr_exact = fem.Expression(
            dt_ue,
            self.fes_u2_0.element.interpolation_points(),
        )
        ut_exact.interpolate(ut_expr_exact)

        u_diff.x.array[:] = np.abs(u_exact.x.array[:] - self.u1_0.x.array[:])
        ut_diff.x.array[:] = np.abs(ut_exact.x.array[:] - self.u2_0.x.array[:])
        print(" u_diff.x.array =", u_diff.x.array)

        u_diff.name = "u-abserr"
        ut_diff.name = "ut-abserr"
        with io.XDMFFile(self.msh.comm, f"{name}-u.xdmf", "w") as xdmf:
            xdmf.write_mesh(self.msh)
            xdmf.write_function(u_diff)
        with io.XDMFFile(self.msh.comm, f"{name}-ut.xdmf", "w") as xdmf:
            xdmf.write_mesh(self.msh)
            xdmf.write_function(ut_diff)
