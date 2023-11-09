"""Determine whether we have a pypardiso installation and use as preferred if found."""

try:
    import pypardiso  # fmt: skip

    SOLVER_TYPE = "pypardiso"
except ImportError:
    pypardiso = None
    SOLVER_TYPE = "petsc-LU"
