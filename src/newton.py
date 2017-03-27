import numpy as np


def newton_step(y, f, fy, fyy, ltol=1e-3):
    '''
    Compute the Newton step for a function fun on the Grassmann
    manifold `Gr(n,p)`.

    Parameters
    ----------
    y : (n, p) ndarray
        Starting point on `Gr(n,p)`.
    f : double
        Value of `fun` at `y`.
    fy : (n, p) ndarray
        Gradient of `fun` at `y` with respect to each element in `y`.
    fyy :  (np, np) ndarray
        Hessian of `fun` at `y` with respect to each element in the
        input matrix. The element in place `((j-1)n + i,(l-1)n + k)`
        should be `d^2f/dy_{ij}dy_{kl}`.
    ltol : double
        Smallest eigenvalue of the Hessian that is tolerated. If the
        Hessian has an eigenvalue smaller than this, a multiple of the
        identity matrix is added to the Hessian so that the resulting
        matrix has `ltol` as its smallest eigenvalue.

    Returns
    -------
    v : (n, p) ndarray
        A vector in the tangent space of Gr(n, p) at `y`, i.e. `y'v = 0.`

    See Also
    --------
    minimize : Minimizing a function over a Gr(n,p), using Newton's
    method if a Hessian is provided, and the steepest descent method
    otherwise.

    Notes
    -----
    The Newton step is `v = Pi(B^-1 grad f)`, where `Pi` is projection
    onto the tangent plane, `B = Hess(f) + cI, Hess(f)` is a matrix
    interpretation of the Hessian of fun on `Gr(n,p)`, `c` is chosen such
    that the smallest eigenvalue of `B` is `ltol` and `grad f` is the gradient
    of `fun` along `Gr(n,p)`.

    References
    ----------
    Edelman, A., Arias, T. A., and Smith S. T. (1998) The geometry of
    algorithms with orthogonality constraints.
    SIAM J. Matrix Anal. Appl., 20(2), 303-353.
  '''
    n, p = y.shape

    pit = np.eye(n) - np.dot(y, y.T)  # projection onto tangent space
    pitTpit = np.dot(pit.T, pit)

    b = -np.dot(pit, fy).reshape(n*p, 1, order='F')

    A = np.zeros((n*p, n*p))

    B = np.dot(y.T, fy)
    B = (B + B.T)*1./2  # Ensure that the Hessian is symmetric

    for j in range(p):
        for l in range(p):
            flj = fyy[j*n:(j+1)*n, l*n:(l+1)*n]
            a1 = np.dot(pit.T, np.dot(flj, pit))
            a2 = B[l, j] * pitTpit
            A[j*n:(j+1)*n, l*n:(l+1)*n] = a1 - a2

    lam = np.min(np.linalg.eigvals(A))
    if (lam <= ltol):
        A = A + np.diag(np.repeat(ltol - lam, n*p))

    v = np.linalg.solve(A, b)
    v = v.reshape(n, p, order='F')
    v = np.dot(pit, v)
    return v
