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
'''
'''

from casadi_extras import Pdq, SystemTrajectory
import unittest
import numpy as np
import numpy.testing as nptest


class TestSystemTrajectory(unittest.TestCase):

    def test_interpolation(self):
        t = [0.1, 0.2, 0.3, 0.4]
        
        x  = np.atleast_2d([2, 3, 4, 5])
        z = np.atleast_2d([4, 5, 6])
        u = np.atleast_2d([-1, -2, -3])
        
        st = SystemTrajectory(x, z, u, Pdq(t, poly_order=1))

        nptest.assert_allclose(st.state(t), x)
        nptest.assert_allclose(st.algState(t[: -1]), z)

        nptest.assert_allclose(st.state([0.15, 0.25]), np.atleast_2d([2.5, 3.5]))
        nptest.assert_allclose(st.algState([0.15, 0.25]), np.atleast_2d([4, 5]))

    
if __name__ == "__main__":
    unittest.main()