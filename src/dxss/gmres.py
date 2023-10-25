from petsc4py import PETSc

from dxss.space_time import SpaceTime


# Custom preconditioner.
class PreTimeMarchingImproved:
    def setUp(self, pc):
        A_shell, _ = pc.getOperators()  # noqa: N806 | convention for matrix
        self.context = A_shell.getPythonContext()  # retrieve & set context

    def apply(self, pc, x, y):  # noqa: ARG002 | 'pc' argument is required by PETSc
        return self.context.st.pre_time_marching_improved(x, y)


def convergence_monitor(ksp, its, rnorm):  # noqa: ARG001
    PETSc.Sys.Print(f"GMRes iteration {its:>3}, residual = {rnorm:4.2e}")


def shellmult(self, A, vec_in, vec_out):  # noqa: N803, ARG001
    self.st.apply_spacetime_matrix(vec_in, vec_out)


def get_gmres_solution(
    A,  # noqa: N803 | convention: Ax = b
    b,
    pre=None,  # noqa: ARG001 | we use a custom preconditioner using PETSc's shell matrix functionality
    x=None,
    maxsteps=100,
    tol=None,
    restart=None,
    printrates=True,
    reltol=None,
):
    # QUESTION: The 'A' passed into this function is an instance of 'SpaceTime.FMatrix'. This serves as the PETSc 'context' for the shell matrix subsequently initialised. The sole reason why this object is copied to the variable named 'context' is simply for user's comprehension. Not sure if this is efficient.
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
    pc = ksp.pc
    pc.setType(pc.Type.PYTHON)
    pc.setPythonContext(PreTimeMarchingImproved())
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
