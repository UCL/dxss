from dxss.solve import get_solve_problem_default_kwargs, solve_problem


def test_solve_smoke():
    """Just execute the main function in solve_1d.py.

    Will fail if any execptions are raised or any errors occur.
    """
    solve_problem(**get_solve_problem_default_kwargs(dimension=1), measure_errors=True)
