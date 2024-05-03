import numpy as np
import pytest
from petsc4py import PETSc

from dxss.gmres import get_gmres_solution


class IdentityPETScMatrix:
    def mult(self, _: PETSc.Mat, vec_in: PETSc.Vec, vec_out: PETSc.Vec) -> None:
        vec_out.array[:] = vec_in.array


class IdentityPlusCircularDiffPETScMatrix:
    def mult(self, _: PETSc.Mat, vec_in: PETSc.Vec, vec_out: PETSc.Vec) -> None:
        vec_out.array[:] = vec_in.array + (
            np.roll(vec_in.array, 1) - np.roll(vec_in.array, -1)
        )


class IdentityPETScPreconditioner:
    def apply(self, _: PETSc.PC, vec_in: PETSc.Vec, vec_out: PETSc.Vec) -> None:
        vec_out.array[:] = vec_in.array


@pytest.mark.parametrize("dimension", [1, 2, 5, 10])
@pytest.mark.parametrize(
    "matrix",
    [IdentityPETScMatrix(), IdentityPlusCircularDiffPETScMatrix()],
)
def test_get_gmres_solution(rng, dimension, matrix):
    """Test GMRes solution `x` for `A @ x = b`."""
    rhs_vector = PETSc.Vec()
    rhs_vector.createSeq(dimension)
    rhs_vector.array[:] = rng.standard_normal(dimension)
    matrix_mult_solution_vector = PETSc.Vec()
    matrix_mult_solution_vector.createSeq(dimension)
    preconditioner = IdentityPETScPreconditioner()
    (solution_vector, _) = get_gmres_solution(matrix, rhs_vector, preconditioner)
    matrix.mult(None, solution_vector, matrix_mult_solution_vector)
    assert np.allclose(matrix_mult_solution_vector, rhs_vector)
