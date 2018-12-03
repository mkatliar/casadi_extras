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
import unittest
import numpy.testing as nptest
import numpy as np
import casadi as cs
import casadi_extras as ce

from casadi_extras.collocation import PiecewisePoly, PolynomialBasis

import matplotlib.pyplot as plt


class PiecewisePolyTest(unittest.TestCase):

    def test_call(self):
        '''Test evaluation of piecewise-polynomial function.
        '''
        pp = PiecewisePoly([10, 20], [42], PolynomialBasis([0]))
        nptest.assert_equal(pp([-1, 10, 15, 20, 100]), np.atleast_2d([42, 42, 42, 42, 42]))

        pp = PiecewisePoly([-1, 2], [[1, 3]], PolynomialBasis([1/3, 2/3]))
        nptest.assert_equal(pp([-1, 0, 1, 2]), np.atleast_2d([-1, 1, 3, 5]))

        pp = PiecewisePoly([-1, 2], [[[1, 3], [10, 30]]], PolynomialBasis([1/3, 2/3]))
        nptest.assert_equal(pp([-1, 0, 1, 2]), np.atleast_2d([[-1, 1, 3, 5], [-10, 10, 30, 50]]))

        pp = PiecewisePoly([-10, 10], [[1, 0, 1]], PolynomialBasis([0, 0.5, 1]))
        nptest.assert_equal(pp([-10, 0, 10]), np.atleast_2d([1, 0, 1]))

        pp = PiecewisePoly([-10, 10, 30], [[1, 0, 1], [2, 0, 2]], PolynomialBasis([0, 0.5, 1]))
        nptest.assert_equal(pp([-10, 0, 10, 20, 30]), np.atleast_2d([1, 0, 2, 0, 2]))

        pp = PiecewisePoly([-10, 10, 30], [[1, 0, 1], [2, 0, 2]], PolynomialBasis([0, 0.5, 1]))
        nptest.assert_equal(pp([-10, 0, 10, 20, 30], continuity='left'), np.atleast_2d([1, 0, 1, 0, 2]))


    def test_discretize(self):
        '''Test discretization of piecewise-polynomial function.
        '''
        pp = PiecewisePoly([-10, 10, 30], [[[1, 0, 1], [1, 2, 3]], [[2, 0, 2], [1, 2, 6]]], PolynomialBasis([0, 0.5, 1]))
        x, y = pp.discretize(0.5)

        plt.plot(x, y.T, '.-')
        plt.grid(True)
        plt.show()


if __name__ == '__main__':
    unittest.main()