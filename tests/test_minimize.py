import numpy as np
import sys
import unittest

sys.path.append('..')
from src import minimize


class testMinimize(unittest.TestCase):

    def test_minimize(self):
        n, p = 20, 4

        A = np.random.rand(n, n)
        A = (A + A.T)/2

        def f1(y):
            return np.sum(np.diag(np.dot(np.dot(y.T, A), y)))*1./2

        def f1y(y):
            return np.dot((A + A.T), y)*1./2

        def f1yy(y):
            B = np.zeros((n*p, n*p))
            for j in range(p):
                B[j*n:(j+1)*n, j*n:(j+1)*n] = A
            return B

        y0 = np.vstack([np.eye(p), np.zeros((n-p, p))])
        opt_res = minimize(y0, f1, f1y, f1yy)
        optval = np.sum(np.sort(np.linalg.eigvals(A))[:p])/2
        self.assertTrue(np.isclose(opt_res['value'], optval))

if __name__ == '__main__':
    unittest.main()
