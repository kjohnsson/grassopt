import numpy as np

from .exceptions import NotTangentVectorError


def geodesic_move(y, v, r=1):
    """
    Move from a point on the Grassmann manifold `Gr(n,p)` represented
    by an n x p matrix `y` along the geodesic in the direction `v`.

    Parameters
    ----------
    y : (n, p) ndarray
        Point where the geodesic starts.
    v : (n, p) ndarray
        Vector in the tangent space of `Gr(n,p)` at `y`, the direction
        of the geodesic.
    r : double
        Step size.

    Returns
    -------
    (n, p) ndarray.
        The new point on `Gr(n,p)`.

    Raises
    ------
    NotTangentVectorError
        If `v` is not the tangent space of `Gr(n,p)` at `y`, i.e. that
        `y'v` does not equal 0.

    Notes
    -----
    Each point on the Grassmannian is represented by an `n x p`
    matrix `y`. `y` should be orthonormal, i.e. `y'y = I`. The columns
    in `y` together form an orthonormal basis for the `p`-dimensional
    subspace of `Rn` that is intended.

    The update is done according to
        `y(r) = [yV  U] [cos(Sigma r)  sin(Sigma r)]' V'`
    where `v = USV'` is the compact singular value decomposition of `v`.

    References
    ----------
    Edelman, A., Arias, T. A., and Smith S. T. (1998) The geometry of
    algorithms with orthogonality constraints.
    SIAM J. Matrix Anal. Appl., 20(2), 303-353.
    """

    if np.max(np.abs(np.dot(y.T, v))) > 1e-6:
        raise NotTangentVectorError

    U, sigma, V = np.linalg.svd(v, full_matrices=False)
    V = V.T
    return np.dot(
        np.hstack([np.dot(y, V), U]), np.dot(
            np.vstack([np.diag(np.cos(sigma*r)),
                       np.diag(np.sin(sigma*r))]),
            V.T))
