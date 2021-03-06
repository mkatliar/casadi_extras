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
"""
Collocation method
"""
import numpy as np
import casadi as cs
import casadi_extras as ce


def collocationPoints(order, scheme):
    '''Obtain collocation points of specific order and scheme.
    '''
    if scheme == 'chebyshev':
        return np.array([0]) if order == 0 else (1 - np.cos(np.pi * np.arange(order + 1) / order)) / 2
    else:
        return np.append(0, cs.collocation_points(order, scheme))


class PolynomialBasis(object):
    '''Polynomial basis.
    '''
    
    def __init__(self, tau):
        '''Make basis of Lagrange polynomials with given roots.

        @param tau roots of the polynomials
        '''

        tau = np.atleast_1d(tau)
        assert tau.ndim == 1
        assert len(tau) > 0

        n = tau.size
        p = []

        for j in range(n):
            # Construct Lagrange polynomials to get the polynomial basis at the collocation point
            pp = np.poly1d([1])
            for r in range(n):
                if r != j:
                    pp *= np.poly1d([1, -tau[r]]) / (tau[j] - tau[r])

            p.append(pp)

        # Calculate lambdas for barycentric representation
        lam = 1 / np.array([np.prod([tau[k] - tau[np.arange(n) != k]]) for k in range(n)])

        # Construct differentiation matrix.
        #
        # The naive method, which has big error for large n, looks like this:
        # D = np.atleast_2d(np.hstack([np.atleast_2d(np.polyder(pp)(tau)).T for pp in p]))
        #
        # We use a better method from here: http://richard.baltensp.home.hefr.ch/Publications/3.pdf
        
        D = np.zeros((n, n))
        for j in range(n):
            for k in range(n):
                if j != k:
                    D[j, k] = lam[k] / lam[j] / (tau[j] - tau[k])
                else:
                    D[j, k] = -np.sum([lam[i] / lam[j] / (tau[j] - tau[i]) for i in range(n) if i != j])

        self.poly = p
        self.D = D
        self.tau = np.array(tau)
        self._lam = lam

    
    @property
    def numPoints(self):
        return len(self.tau)


    @property 
    def order(self):
        '''Order of basis polynomials
        '''
        return len(self.tau) - 1


    def interpolationMatrix(self, t):
        '''Interpolation matrix for specified interpolation points.

        @param t interpolation points
        '''

        t = np.atleast_1d(t)    # Convert to numpy type s.t. the indexing below works
        assert np.ndim(t) == 1 
        
        r = []
        for xi in t:
            # Check if xi is in tau
            ind = xi == self.tau

            if np.any(ind):
                # At a node
                assert np.sum(ind) == 1
                y = ind.astype(float)
            else:
                # Between nodes
                y = (self._lam / (xi - self.tau)) / np.sum(self._lam / (xi - self.tau))

            r.append(y)

        return np.vstack(r)


    def compose(self, c):
        '''Make polynomials from their decomposition in the basis.

        @param c matrix of basis coefficients of size M-by-N.
        @return the list of resulting polynomials, which are
        $q_i=\sum_{j=0}^{N-1} c_{i,j} p_j$ where i=0..M.
        '''
        M, N = np.shape(c)
        q = []
        
        for i in range(M):
            q_i = np.poly1d([0])
            for j in range(N):
                q_i += c[i, j] * self.poly[j]
            q.append(q_i)

        return q



def polynomialInterpolator(x):
    """Polynomial interpolator with nodes at x
    """

    assert x.ndim == 1
    N = x.size - 1
    n = np.atleast_2d(np.arange(N + 1)).T
    X = x ** n

    return lambda u, xx: np.dot(u, np.linalg.solve(X, xx ** n))


def barycentricInterpolator(x):
    """Barycentric interpolator with nodes at x
    """

    basis = PolynomialBasis(x)

    def p(u, xq):
        return np.dot(u, basis.interpolationMatrix(xq).T)

    return p


def cheb(N, t0, tf):
    """Chebyshev grid and differentiation matrix

    The code is based on cheb.m function from the book
    "Spectral Methods in MATLAB" by Lloyd N. Trefethen 
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.473.7647&rep=rep1&type=pdf

    @param N order of collocation polynomial.
    @param t0 left end of the collocation interval
    @param t0 right end of the collocation interval

    @return a tuple (D, t) where D is a (N+1)-by-(N+1) differentiation matrix 
    and t is a vector of points of length N+1 such that t[0] == t0 and t[-1] == tf.

    TODO: deprecate?
    """
    
    tau = collocationPoints(N, 'chebyshev')
    basis = PolynomialBasis(tau)
    return basis.D / (tf - t0), tau * (tf - t0) + t0
