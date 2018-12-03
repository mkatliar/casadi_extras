##
## Copyright (c) 2018 Mikhail Katliar.
## 
## This file is part of CasADi Extras 
## (see https://github.com/mkatliar/casadi_extras).
## 
## This program is free software: you can redistribute it and/or modify
## it under the terms of the GNU Lesser General Public License as published by
## the Free Software Foundation, either version 3 of the License, or
## (at your option) any later version.
## 
## This program is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU Lesser General Public License for more details.
## 
## You should have received a copy of the GNU Lesser General Public License
## along with this program. If not, see <http://www.gnu.org/licenses/>.
##
import numpy as np
import casadi as cs


class ButcherTableau(object):
    
    def __init__(self, A, b, c):
        assert np.ndim(A) == 2
        assert A.shape[0] == A.shape[1]

        assert np.ndim(b) == 1
        assert np.size(b) == A.shape[1]

        assert np.ndim(c) == 1
        assert np.size(c) == A.shape[0]

        self._A = np.atleast_2d(A)
        self._b = np.atleast_1d(b)
        self._c = np.atleast_1d(c)
        

    @property
    def A(self):
        return self._A


    @property
    def b(self):
        return self._b


    @property
    def c(self):
        return self._c


def butcherTableuForCollocationMethod(order, method):

    tau = cs.collocation_points(order, method)
    A = np.zeros((order, order))
    b = np.zeros(order)
    
    for j in range(order):
        # Construct Lagrange polynomials to get the polynomial basis at the collocation point
        p = np.poly1d([1])
        for r in range(order):
            if r != j:
                p *= np.poly1d([1, -tau[r]]) / (tau[j] - tau[r])

        # Evaluate the integral of the polynomial to get the coefficients of the quadrature function
        pint = np.polyint(p)
        A[:, j] = pint(tau)
        b[j] = pint(1.0)

    return ButcherTableau(A=A, b=b, c=tau)