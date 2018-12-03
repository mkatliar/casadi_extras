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

from casadi_extras.collocation.butcher import *


class ButcherTest(unittest.TestCase):

    def test_ctor(self):
        tableau = ButcherTableau(A=np.random.rand(3, 3), b=np.random.rand(3), c=np.random.rand(3))


    def test_butcherTableuForCollocationMethod(self):
        '''
        Test correctness of Butcher tableus for collocation methods.
        '''

        for d in range(1, 5):
            tableau = butcherTableuForCollocationMethod(d, 'legendre')
            
            # Test polynomial and its integral
            p = np.poly1d(np.random.rand(d))
            P = np.polyint(p)
            
            nptest.assert_allclose(np.dot(tableau.A, p(tableau.c)), P(tableau.c))
            nptest.assert_allclose(np.dot(tableau.b, p(tableau.c)), P(1.0))


if __name__ == '__main__':
    unittest.main()