try:
    from dxss.solve_1d import solve_problem
except ImportError:
    from dxss.solve_1d import SolveProblem as solve_problem  # noqa: N813


def test_solve_1d():
    """Execute the main solver function in solve_1d.py.

    Acts partially as a smoke test: this will fail if any exceptions are raised
    or errors occur in the solbing code. But also verify that the resulting
    errors are below a threshold to ensure we recover the expected (damped sine
    wave) solution.
    """
    _, errors = solve_problem(measure_errors=True)
    assert errors["L-infty-L2-error-u"] < 0.1
    assert errors["L-infty-L2-error-ut"] < 1.0
    assert errors["L2-L2-error-u_t"] < 1.0
