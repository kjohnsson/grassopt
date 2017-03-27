import numpy as np
import matplotlib.pyplot as plt

from grassopt import minimize

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

fig, axs = plt.subplots(1, 2, figsize=(12, 4))
fy = [f1(yy) for yy in opt_res['y']]
axs[0].plot(range(len(fy)), fy)
axs[0].hlines(optval, 0, len(fy)-1)
axs[0].set_ylabel('f(Y)')
axs[0].set_xlabel('Iteration')

err = abs(fy - optval)
axs[1].semilogy(range(len(fy)), err)
axs[1].set_ylabel('Absolute error')
axs[1].set_xlabel('Iteration')
plt.show()