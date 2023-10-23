from math import sqrt


def givens_rotation(v1: float, v2: float) -> tuple[float, float]:
    """
    Computes and returns the coefficients of the rotation matrix used to perform a Givens rotation.

    # TODO: can the Givens rotation def, and application functions be
    # moved out of this into a general utilities library?

    Args:
        v1: The first component of the vector.
        v2: The second component of the vector, which is to be zeroed out

    Returns:
        The coefficients of the rotation matrix as a tuple (cs, sn)
    """
    if v2 == 0:
        if v1 == 0:
            msg = "v1 & v2 cannot both be zero for Givens rotation"
            raise ValueError(msg)
        return 1, 0
    if v1 == 0:
        return 0, v2 / abs(v2)
    if abs(v1) > abs(v2):
        t = v2 / v1
        cs = 1.0 / sqrt(1 + t**2.0)
        sn = t * cs
    else:
        tau = v1 / v2
        sn = 1.0 / sqrt(1 + tau**2.0)
        cs = tau * sn
    return cs, sn
