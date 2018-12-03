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
import casadi_extras as ce

from .polynomial import PolynomialBasis


class PiecewisePoly(object):
    """Piecewise-polynomial function.
    """

    def __init__(self, intervals, coefficients, basis):
        """Constructor

        @param intervals list of N+1 points in ascending order defining N intervals.
        @param coefficients list of N matrices, each of them defining basis coefficients on one interval.
        @param basis PolynomialBasis defined on interval [0, 1]
        """

        assert len(intervals) > 1
        N = len(intervals) - 1

        assert len(coefficients) == N

        self._basis = basis
        self._intervals = intervals
        self._coefficients = coefficients


    @property
    def intervals(self):
        """Interval bounds"""
        return self._intervals


    @property
    def numIntervals(self):
        '''Number of intervals'''
        return len(self._intervals) - 1


    def __call__(self, t, continuity='right'):
        """Evaluate piecewise-polynomial values at given points.

        @param t points at which to calculate values. Can be a scalar of a vector.
        @param continuity specifies which value to return at interval boundaries:
        - 'left' means the value from the left interval
        - 'right' means the value from the right interval
        """

        if continuity not in ['left', 'right']:
            raise ValueError('Invalid "continuity" value {0} in PiecewisePoly.__call__()'.format(continuity))


        l = []
        
        for ti in np.atleast_1d(t):
            # interval index
            i = np.clip(np.searchsorted(self._intervals, ti, continuity) - 1, 0, len(self._intervals) - 2)

            # query point value transformed to [0, 1] interval
            tau = (ti - self._intervals[i]) / (self._intervals[i + 1] - self._intervals[i])

            # interpolate point
            l.append(np.dot(self._coefficients[i], self._basis.interpolationMatrix(tau).T))

        return np.atleast_2d(np.hstack(l))


    def discretize(self, step):
        '''Return x and y values, such that x is sampled at steps not greater than step,
        and y are the function values at x. For each interval, both its ends are always included in x.

        The returned values are useful for plotting the function.
        '''

        x = []
        y = []

        for i in range(self.numIntervals):
            tau = np.arange(0, 1, step / (self._intervals[i + 1] - self._intervals[i]))
            if tau[-1] != 1:
                assert tau[-1] < 1
                tau = np.append(tau, 1)

            x.append(self._intervals[i] + tau * (self._intervals[i + 1] - self._intervals[i]))
            y.append(np.dot(self._coefficients[i], self._basis.interpolationMatrix(tau).T))

        return np.hstack(x), np.atleast_2d(np.hstack(y))
