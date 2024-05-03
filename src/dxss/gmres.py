from __future__ import annotations

from petsc4py import PETSc


def convergence_monitor(ksp: PETSc.KSP, its: int, rnorm: float) -> None:  # noqa: ARG001
    """A simple rate-printing convergence monitor.

    The function trace is dictated by PETSc.
    https://petsc.org/release/petsc4py/reference/petsc4py.typing.KSPMonitorFunction.html

    Args:
        ksp: The PETSc.KSP instance. Currently unused (ARG001). Needed to comply
             with PETSc's monitor function format.
        its: The iteration count passed in by PETSc.
        rnorm: The (estimated) 2-norm of (preconditioned) residual.

    Note: The `rnorm` is _not_ the relative normalisation despite its name.
    """
    PETSc.Sys.Print(f"GMRes iteration {its:>3}, residual = {rnorm:4.2e}")


def get_gmres_solution(
    A,  # noqa: N803 | convention: Ax = b
    b: PETSc.Vec,
    pre,
    maxsteps: int = 100,
    tol: float | None = None,
    restart: int | None = None,
    printrates: bool = True,
    reltol: float | None = None,
) -> tuple[PETSc.Vec, PETSc.Vec]:
    """Solve linear system `A @ x = b` with generalized minimal residual method (GMRes).

    Uses PETSc GMRes implementation.

    Args:
        A: Object implementing PETSc `MatPythonProtocol` to use as matrix `A`.
        b: PETSc vector object to use as right-hand side term `b`.
        pre: Object implementing `PCPythonProtocol` to use as GMRes preconditioner.
        maxsteps: Maximum number of GMRes iterations to use.
        tol: Absolute convergence tolerance - absolute size of the (preconditioned)
            residual norm.
        restart: Number of iterations after which to restart.
        printrates: Whether to show convergence statistics during solving.
        reltol: Relative convergence tolerance, relative decrease in the
            (preconditioned) residual norm.
    """
    # The 'A' passed into this function serves as the PETSc 'context' for the shell matrix.
    # Renaming for readability and consistency with PETSc docs. Potentially a minor inefficiency.
    context = A

    N = b.getSize()  # noqa: N806 | global vector size variables use uppercase in PETSc

    # Create a PETSc shell matrix of appropriate global size (N x N)
    A_shell = PETSc.Mat().createPython(N, context)  # noqa: N806 | convention for matrix
    A_shell.setUp()

    ksp = PETSc.KSP()
    ksp.create(comm=A_shell.getComm())
    ksp.setOperators(A_shell)  # set the linear operator(s) for the KSP object

    ksp.setType(PETSc.KSP.Type.GMRES)  # note: default orthogonisation uses classical GS

    if restart is None:
        restart = 1000  # Currently set to a high number for faster convergence
    ksp.setGMRESRestart(restart)  # set number of iterations until restart

    if tol is None:
        tol = 1e-12
    if reltol is None:
        reltol = 1e-7
    ksp.setTolerances(rtol=reltol, atol=tol, max_it=maxsteps)

    # Prepare shell preconditioner
    pc = ksp.pc
    pc.setType(pc.Type.PYTHON)
    pc.setPythonContext(pre)
    pc.setUp()

    petsc_opts = PETSc.Options()
    petsc_opts.setValue("ksp_gmres_cgs_refinement_type", "refine_ifneeded")
    petsc_opts.setValue("ksp_gmres_modifiedgramschmidt", True)  # This is important!
    petsc_opts.setValue("ksp_gmres_preallocate", True)  # efficiency vs memory
    ksp.setFromOptions()  # allow the user to set command-line options at runtime for tuning the solver

    if printrates:
        ksp.setMonitor(convergence_monitor)

    ksp.view()
    ksp.setConvergenceHistory()

    x = A_shell.createVecRight()  # sol vec conforming to matrix partitioning

    ksp.solve(b, x)  # attempt to iterate to convergence
    ksp_reason = ksp.getConvergedReason()  # check for convergence
    ksp_iter = ksp.getIterationNumber()  # iterations completed

    PETSc.Sys.Print(f"The convergence reason code is {ksp_reason}")
    if ksp_reason < 0:
        PETSc.Sys.Print(f"Solver did not converge after {ksp_iter} iterations.")
        raise PETSc.Error

    r = ksp.buildResidual()

    return (x, r)
