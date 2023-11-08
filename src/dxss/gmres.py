from petsc4py import PETSc

from dxss.space_time import SpaceTime


class PreTimeMarchingImproved:
    """Class that wraps our research-grade improved time marching precondition.

    The class wrapper is needed to get it into the format needed by PETSc's KSP
    interface.

    Attributes:
        context: (dxss.SpaceTime) the sparse matrix context containing n, T, ...

    Todo:
        Ultimately we could move the code really into this class, and
        potentially make some computational savings by splitting the function
        SpaceTime.pre_time_marching_improved into this class's setup and apply.
        Also TODO is to review this docstring.
    """

    def setUp(self, pc):
        A_shell, _ = pc.getOperators()  # noqa: N806 | convention for matrix
        self.context = A_shell.getPythonContext()  # retrieve & set context

    def apply(self, pc, x, y):  # noqa: ARG002 | 'pc' argument is required by PETSc
        return self.context.st.pre_time_marching_improved(x, y)


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


def shellmult(
    self,
    A: SpaceTime.FMatrix,  # noqa: ARG001,N803
    vec_in: PETSc.Vec,
    vec_out: PETSc.Vec,
) -> None:
    """Shell matrix multiplication wrapper function.

    Needed to monkey-patch an instance of dxss.SpaceTime to recover the desired
    behavior.

    Args:
        A: The shell representation of the LHS in our 'matrix-free' scheme.
        vec_in: The vector to multiply with A.
        vec_out: The output vector to contain the multiplication result.

    Todo:
        Ultimately this should be integrated into the dxss.SpaceTime class
        during a refactoring pass.
    """
    self.st.apply_spacetime_matrix(vec_in, vec_out)


def get_gmres_solution(
    A,  # noqa: N803 | convention: Ax = b
    b,
    pre=None,
    x=None,
    maxsteps=100,
    tol=None,
    restart=None,
    printrates=True,
    reltol=None,
):
    # The 'A' passed into this function serves as the PETSc 'context' for the shell matrix.
    # Renaming for readability and consistency with PETSc docs. Potentially a minor inefficiency.
    context = A

    # Patch the 'mult' method only for this 'context' instance of SpaceTime.FMatrix to comply with PETSc's signature
    context.mult = shellmult.__get__(context, SpaceTime.FMatrix)

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
    if pre is None:
        pre = PreTimeMarchingImproved()
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

    # Overwrite the supplied placeholder solution vector (but retain the function argument for API comptabiility)
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
