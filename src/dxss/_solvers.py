from typing import Any

from petsc4py import PETSc


def get_lu_solver(msh: Any, mat: Any) -> Any:
    """Create a PETSc KSP solver with LU preconditioner.

    Args:
        msh: The mesh object.
        mat: The matrix object.

    Returns:
        The PETSc KSP solver with LU preconditioner.

    Todo:
        Correct the type annotations. Currently, the return type is Any but it
        should be a PETSc.Mesh?
    """
    solver = PETSc.KSP().create(msh.comm)
    solver.setOperators(mat)
    solver.setType(PETSc.KSP.Type.PREONLY)
    solver.getPC().setType(PETSc.PC.Type.LU)
    return solver


class PySolver:
    """A solver class for use with the Pardiso solver.

    Attributes:
        Asp: The sparse matrix object, A.
        solver: The Pardiso solver.
    """

    def __init__(self, Asp, psolver):  # noqa: N803 | convention Ax = b
        self.Asp = Asp
        self.solver = psolver

    def solve(
        self,
        b_inp: list[float],
        x_out: list[float],
        set_phase: bool = True,
    ) -> None:
        """
        Solve the linear system Ax = b using the Pardiso solver.

        Args:
            b_inp: The input vector b.
            x_out: The output vector x.
            set_phase: Should we set the phase of the solver whilst setting up?

        Todo:
            Check the type annotations with Janosch.
        """
        self.solver._check_A(self.Asp)
        b = self.solver._check_b(self.Asp, b_inp)
        if set_phase:
            self.solver.set_phase(33)
        x_out[:] = self.solver._call_pardiso(self.Asp, b)[:]