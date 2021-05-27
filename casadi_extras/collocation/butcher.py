##
## Copyright (c) 2018-2021 Mikhail Katliar.
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
## You should have received a copy of the GNU Lesser General Public Licenseдзеўкамі
## along with this program. If not, see <http://www.gnu.org/licenses/>.
##
import numpy as np
import casadi as cs
from deprecated import deprecated


class ButcherTableau(object):
    """Butcher tableau for Runge-Kutta integration method."""

    def __init__(self, A: np.array, b: np.array, c: np.array):
        """Constructor

        @param A NxN Runge–Kutta matrix
        @param b vector of weights of size N
        @param a vector of nodes of size N
        """
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

    def __repr__(self):
        return 'ButcherTableau(A={}, b={}, c={})'.format(self._A, self._b, self._c)


def butcherTableuForCollocationPoints(tau: np.array):
    """Butcher table for collocation method defined by a set of nodes

    @param tau collocation nodes
    """
    order = len(tau)
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


@deprecated(version='1.0.4', reason='Use butcherTableuForCollocationPoints() instead')
def butcherTableuForCollocationMethod(order: int, method: str):
    """Butcher table for collocation method with a given number of collocation points

    @param order number of collocation points
    @param method defines collocation method ('radau' or 'legendre')
    """

    return butcherTableuForCollocationPoints(cs.collocation_points(order, method))
