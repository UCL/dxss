import sys
import warnings

from petsc4py import init

init(sys.argv)  # allow the parsing of command-line options
from petsc4py import PETSc  # noqa: E402

# Check that PETSc has been initialised. If not, give up!
if not PETSc.Sys.isInitialized():
    PETSc.Sys.Print("PETSc did not initialise successfully")
    raise PETSc.Error


comm = PETSc.COMM_WORLD  # initialise the PETSc communicator for use by PETSc objects.


def get_gmres_solution(
    A,  # noqa: N803 | convention: Ax = b
    b,
    maxsteps=100,
    pre=None,  # TODO: to be implemented
    printrates=False,  # TODO: to be implemented
):
    if printrates:
        warnings.warn(
            "Print-monitoring of convergence rates is yet to be incorporated in our current invocation of PETSc's GMRES implementation.",
            stacklevel=1,
        )

    if pre:
        warnings.warn(
            "Ability to choose a custom Preconditioner is yet to be incorporated in our current invocation of PETSc's GMRES implementation.",
            stacklevel=1,
        )

    ksp = PETSc.KSP()
    ksp.create(comm=A.getComm())
    ksp.setOperators(A)  # set the linear operator(s) for the KSP object
    ksp.setType(PETSc.KSP.Type.GMRES)  # use the GMRES method

    # TODO: Preconditioner: hard-coding Additive Schwarz Method (asm) for now. Will need to be replaced with the custom function from space_time.py soon
    ksp.getPC().setType(PETSc.PC.Type.ASM)  # TODO: this is important to change!

    ksp.setTolerances(max_it=maxsteps)  # set the maximum number of iterations

    ksp.setFromOptions()  # allow the user to set command-line options at runtime for tuning the solver

    x = A.createVecRight()  # placeholder solution vec conforming to matrix partitioning

    ksp.solve(b, x)  # attempt to iterate to convergence

    ksp_reason = ksp.getConvergedReason()  # check for convergence
    ksp_iter = ksp.getIterationNumber()  # iterations completed

    if ksp_reason < 0:
        PETSc.Sys.Print(
            f"The iterative solver did not converge after {ksp_iter} iterations.",
        )
        raise PETSc.Error

    x.viewFromOptions("-view_sol")

    r = ksp.buildResidual()

    return (x, r)
