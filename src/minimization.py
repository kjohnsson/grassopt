import numpy as np

from .exceptions import LineSearchFailError
from .geodesic import geodesic_move
from .newton import newton_step


def minimize(y0, fun, grEu, hessEu=None, maxit=50, tol=1e-8, ltol=1e-3,
             alpha=0.2, beta=0.5):
    '''
    Minimizing a function over a Gr(n,p), using Newton's method if a
    Hessian is provided, and the steepest descent method otherwise.

    Parameters
    ----------
    y0 : (n, p) ndarray
        Starting point at the Grassmann manifold.
    fun : function
        Function to be minimized. `fun` should take a (n, p) ndarray as
        input and return a scalar.
    grEu : function
        Gradient of `fun` in Euclidean space. `grEu` should take a
        (n, p) ndarray as input and return a (n, p) ndarray.
    hessEu : function
        Hessian of `fun` in Euclidean space. `hessEu` should take a
        (n, p) ndarray as input and return a (np, np) ndarray.
    maxit: integer
        Maximum number of iterations.
    tol : double
      Tolerated value for `(grad f)'B(grad f)` (Newton) or `||grad f||^2`
      (steepest descent) at the optimal point.
    ltol : double
        Smallest eigenvalue of the Hessian that is tolerated. If the
        Hessian has an eigenvalue smaller than this, a multiple of the
        identity matrix is added to the Hessian so that the resulting
        matrix has `ltol` as its smallest eigenvalue.
    alpha : double
        Parameter for line search.
    beta : double
        Parameter for line search.

    Returns
    -------
    Dictionary with items
        value : double
            Value of \code{fun} at the found solution.
        y : list of (n, p) ndarrays
            List with the points on the Grassmann manifold (matrices)
            that has been gone through during the iterations.
        lam2 : list of doubles
            List with the values of `(grad f)'B(grad f)` (Newton) or
            `||grad f||^2` (steepest descent) at the points in `y`.

    Raises
    ------
    LineSearchFailError
        If line search fails.

    Notes
    -----
    The Newton step is `v = Pi(B^-1 grad f)`, where `Pi` is the
    projection onto the tangent plane, `B = Hess(f) + cI`, `Hess(f)` is
    a matrix interpretation of the Hessian of `fun` on `Gr(n,p)`, `c` is
    chosen such that the smallest eigenvalue of `B` is `ltol` and
    `grad f` is the gradient of `fun` along `Gr(n,p)`.

    For the steepest descent method, the corresponding step is `-grad f`,
    i.e. the negative of the gradient along `Gr(n,p)`.

    For both methods, backtracking line search is used.

    References
    ----------
    Edelman, A., Arias, T. A., and Smith S. T. (1998) The geometry of
    algorithms with orthogonality constraints.
    SIAM J. Matrix Anal. Appl., 20(2), 303-353.
    '''
    steepest_descent = hessEu is None

    n = y0.shape[0]
    y = [y0]
    lam2 = -np.ones(maxit)
    yy_next = y0

    for it_main in range(maxit):
        yy = np.copy(yy_next)
        grEuy = grEu(yy)
        gr = np.dot((np.eye(n) - np.dot(yy, yy.T)), grEuy)
        v = -gr if steepest_descent else newton_step(
            yy, fun(yy), grEuy, hessEu(yy), ltol=ltol)
        lam2[it_main] = -np.sum(v*gr)

        if lam2[it_main] < 2*tol:
            break

        r = 1
        it_linesearch = 0
        fyy = fun(yy)
        yy_next = geodesic_move(yy, v, r)

        while fun(yy_next) >= fyy - alpha*r*lam2[it_main]:
            r *= beta
            yy_next = geodesic_move(yy, v, r)
            it_linesearch += 1
            if it_linesearch > 50:
                raise LineSearchFailError

        y.append(yy_next)

    return({'value': fun(yy_next), 'y': y, 'lam2': lam2})
