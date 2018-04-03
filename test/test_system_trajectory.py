'''
'''

from casadi_tools import Pdq, SystemTrajectory
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