"""
Collocation method
"""
import numpy as np
import casadi as cs
import casadi_tools as ct

from .polynomial_basis import PolynomialBasis, collocationPoints


class Pdq(object):
    """Polynomial-based Differential Quadrature (PDQ).

    See "Differential Quadrature and Its Application in Engineering: 
    Engineering Applications, Chang Shu, Springer, 2000, ISBN 978-1-85233-209-9"

    https://link.springer.com/content/pdf/10.1007%2F978-1-4471-0407-0.pdf
    """

    def __init__(self, t, poly_order=5, polynomial_type='chebyshev'):
        """Constructor

        @param t N+1 points defining N collocation intervals in ascending order.
        @param poly_order order of collocation polynomial.
        @param polynomial_type collocation polynomial type.
        """

        N = len(t) - 1

        # Make collocation points vector and differential matrices list
        collocation_points = []
        collocation_groups = []
        basis = []
        k = 0
        interval_index = []

        for i in range(N):
            t_i = collocationPoints(poly_order, polynomial_type) * (t[i + 1] - t[i]) + t[i]
            basis_i = PolynomialBasis(t_i)
            
            assert t_i.shape == (poly_order + 1,)

            # Ensure that the points are from left to right.
            assert np.all(np.diff(t_i) > 0)

            basis.append(basis_i)
            collocation_points.append(t_i)
            collocation_groups.append(np.arange(k, k + basis_i.numPoints))
            interval_index.append(k)
            k += basis_i.numPoints

        interval_index.append(k)

        # Stack collocation points in one vector
        collocation_points = np.hstack(collocation_points)

        self._basis = basis
        self._collocationPoints = collocation_points

        # Indices of collocation points belonging to the same interval, including both ends.
        self._collocationGroups = collocation_groups
        self._intervalBounds = np.array(t)
        self._polyOrder = poly_order
        self._intervalIndex = np.array(interval_index)


    @property
    def collocationPoints(self):
        """Collocation points"""
        return self._collocationPoints


    @property
    def numCollocationPoints(self):
        '''Total number of collocation points'''
        return len(self._collocationPoints)


    @property
    def intervalIndex(self):
        '''Indices corresponding to bounds of collocation intervals.'''
        return self._intervalIndex


    @property
    def intervalBounds(self):
        """Interval bounds"""
        return self._intervalBounds


    @property
    def numIntervals(self):
        '''Number of collocation intervals'''
        return len(self._intervalBounds) - 1


    @property
    def polyOrder(self):
        """Degree of interpolating polynomial"""
        return self._polyOrder


    @property
    def basis(self):
        '''Polynomial bases on each interval'''
        return self._basis


    @property
    def t0(self):
        """The leftmost collocation point"""
        return self._collocationPoints[0]


    @property
    def tf(self):
        """The rightmost collocation point"""
        return self._collocationPoints[-1]


    def intervalLength(self):
        """Distance between the leftmost and the rightmost collocation points

        TODO: deprecate?
        """
        return self._collocationPoints[-1] - self._collocationPoints[0]


    def derivative(self, y):
        """Calculate derivative from function values
        """
        assert y.shape[1] == self.numCollocationPoints
        
        dy = []
        k = 0

        for b in self._basis:
            dy.append(cs.mtimes(y[:, k : k + b.numPoints], b.D.T))
            k += b.numPoints

        return cs.horzcat(*dy)


    def integral(self, dy):
        """Calculate integral from derivative values
        """

        assert dy.shape[1] == self.numCollocationPoints

        y0 = cs.MX.zeros(dy.shape[0])
        y = []
        k = 0
        for i, b in enumerate(self._basis):
            N = b.numPoints - 1

            # Calculate y the equation 
            # dy = cs.mtimes(y[:, 1 :], D[: -1, 1 :].T)
            #invD = cs.inv(D[: -1, 1 :])
            #y.append(cs.transpose(cs.mtimes(invD, cs.transpose(dy[:, k : k + N]))))

            # TODO: this is correct only if both end points are collocation points
            # TODO: there should be a more precise way without dropping one point
            assert np.all(b.tau[[0, -1]] == self._intervalBounds[[i, i + 1]])
            y.append(cs.transpose(cs.solve(b.D[: -1, 1 :], cs.transpose(dy[:, k : k + N]))) + cs.repmat(y0, 1, N))
            k += b.numPoints
            y0 = y[-1][:, -1]

        return cs.horzcat(*y)


    def expandInput(self, u):
        '''Return input at collocation points given the input u on collocation intervals.
        '''

        n = self.numIntervals
        assert u.shape[1] == n
        return cs.horzcat(*[cs.repmat(u[:, k], 1, b.numPoints) for k, b in enumerate(self._basis)])


    def interpolator(self, continuity='both'):
        """Create interpolating function based on values at collocation points

        @param specifies continuity of the interpolated function at the interval boundaries:
        - 'left' means that the function in continuous from the left,
        - 'right' means that the function in continuous from the right,
        - 'both' means that the function is continuous both from the left and from the right.
        """

        groups = self._collocationGroups

        if continuity == 'left':
            side = 'right'
        elif continuity == 'right':
            side = 'left'
        else:
            raise ValueError('Invalid "continuity" value {0} in Pdq.interpolator()'.format(continuity))

        fi_cl = [barycentricInterpolator(b.tau) for b in self._basis]

        def interp(x, t):
            expected_x_cols = self._collocationPoints.size
            if x.shape[1] != expected_x_cols:
                raise ValueError('Invalid number of columns in interpolation point matrix')

            l = []
            
            for ti in np.atleast_1d(t):
                i = np.clip(np.searchsorted(self._intervalBounds, ti, side) - 1, 0, len(self._intervalBounds) - 2)  # interval index
                l.append(fi_cl[i](x[:, groups[i]], ti))

            return np.hstack(l)

        return interp


def polynomialInterpolator(x):
    """Polynomial interpolator with nodes at x"""

    assert x.ndim == 1
    N = x.size - 1
    n = np.atleast_2d(np.arange(N + 1)).T
    X = x ** n

    return lambda u, xx: np.dot(u, np.linalg.solve(X, xx ** n))


def barycentricInterpolator(x):
    """Barycentric interpolator with nodes at x"""

    assert np.ndim(x) == 1
    N = np.size(x) - 1
    n = np.arange(N + 1)
    x = np.atleast_1d(x)    # Convert to numpy type s.t. the indexing below works

    a = [np.prod(x[j] - x[n[n != j]]) for j in n]

    def p(u, xq):
        r = []

        # Convert scalar argument to a vector
        xq = np.atleast_1d(xq)
        
        # Converting to np.array is a workaround for https://github.com/casadi/casadi/issues/2221
        u = np.array(u)

        for xi in xq:
            # Check if xi is in x
            ind = xi == x

            if np.any(ind):
                # At a node
                assert np.sum(ind) == 1
                y = u[:, ind]
            else:
                # Between nodes
                y = np.dot(u, np.atleast_2d(1 / (a * (xi - x))).T) / np.sum(1 / (a * (xi - x)))

            r.append(y)

        return np.hstack(r)

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
