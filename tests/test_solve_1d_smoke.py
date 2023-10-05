from dxss.solve_1d import solve_problem


def test_solve_1d_smoke():
    """Just execute the main function in solve_1d.py.

    Will fail if any execptions are raised or any errors occur.
    """
    solve_problem(measure_errors=True)
