import sys

import numpy as np
from petsc4py import init

init(sys.argv)  # allow the parsing of command-line options
from petsc4py import PETSc  # noqa: E402

from dxss.gmres import get_gmres_solution  # noqa: E402

# Check that PETSc has been initialised. If not, give up!
if not PETSc.Sys.isInitialized():
    PETSc.Sys.Print("PETSc did not initialise successfully")
    raise PETSc.Error


comm = PETSc.COMM_WORLD  # initialise the PETSc communicator for use by PETSc objects.


class TestGMRES:
    def index_to_grid(self, r, n):
        """Convert a row number into a grid point."""
        return (r // n, r % n)

    def test_fivepoint_laplacian_unit_square_zero_dirichlet_bc(self):
        """Solve a constant coefficient Poisson problem on a regular grid."""
        petsc_cli_opts = PETSc.Options()

        n = 5  # default value of n. The matrix will be sized n^2 x n^2
        n = petsc_cli_opts.getInt("n", 5)  # parse cli options to obtain user set n

        h = 1.0 / (n + 1)  # calculate the grid spacing

        # Create a PETSc matrix
        A = PETSc.Mat()  # noqa: N806 | convention: Ax = b
        A.create(comm=PETSc.COMM_WORLD)
        A.setType(PETSc.Mat.Type.AIJ)  # set type to sparse (auto-choose SEQAIJ/MPIAIJ)
        A.setSizes((n * n, n * n))  # set the global size of the matrix as n^2 x n^2
        A.setFromOptions()  # allow matrix properies to be altered via command-line options
        A.setPreallocationNNZ(5)  # max no of non-zeros/row (efficient preallocation)

        (rstart, rend) = A.getOwnershipRange()  # get row ranges local to this process

        # fill the entries of the matrix with values of a standard Laplacian
        for row in range(rstart, rend):
            i, j = self.index_to_grid(row, n)
            A[row, row] = 4.0 / h**2
            if i > 0:
                column = row - n
                A[row, column] = -1.0 / h**2
            if i < n - 1:
                column = row + n
                A[row, column] = -1.0 / h**2
            if j > 0:
                column = row - 1
                A[row, column] = -1.0 / h**2
            if j < n - 1:
                column = row + 1
                A[row, column] = -1.0 / h**2

        # Assemble the distributed matrix across all MPI ranks
        A.assemblyBegin()
        A.assemblyEnd()

        A.viewFromOptions("-view_mat")  # make sparsity pattern viewable via cli option

        b = A.createVecLeft()  # create RHS vector that conforms to matrix partitioning
        b.set(1.0)  # set all values of the RHS vec to 1.0

        x_gmres_petsc_vec = get_gmres_solution(A, b, maxsteps=100)  # gmres iterations
        x_gmres_array = x_gmres_petsc_vec.getArray(readonly=True)
        x_truth_array = np.array(
            [
                0.0264423,
                0.0389956,
                0.0427354,
                0.0389963,
                0.0264421,
                0.0389955,
                0.0590275,
                0.0651707,
                0.0590277,
                0.0389954,
                0.0427351,
                0.0651706,
                0.0721157,
                0.0651711,
                0.042735,
                0.0389962,
                0.0590277,
                0.0651712,
                0.0590275,
                0.0389957,
                0.026442,
                0.0389954,
                0.0427351,
                0.0389959,
                0.0264429,
            ],
        )
        assert np.allclose(x_gmres_array, x_truth_array, atol=1e-6)

        x_gmres_norm = np.linalg.norm(x_gmres_array)
        assert np.allclose(x_gmres_norm, 0.24164846282315805, atol=1e-14)
