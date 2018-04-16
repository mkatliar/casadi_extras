"""
Tests for the inequality module
"""


import unittest
import numpy.testing as nptest
import numpy as np
import casadi as cs
import casadi_tools as ct


class InequalityTest(unittest.TestCase):
    """
    Tests for Inequality class
    """

    def test_ctor(self):
        '''Test Inequality ctor
        '''

        i = ct.Inequality(expr=cs.MX.sym('x', 2), lb=np.array([-1, -2]), ub=np.array([3, 4]))

        nptest.assert_equal(i.lb, np.array(cs.DM([-1, -2])))
        nptest.assert_equal(i.ub, np.array(cs.DM([3, 4])))


    def test_ctorInconsistentShape(self):
        '''Test Inequality ctor with arguments of inconsistent shape
        '''

        self.assertRaises(Exception, ct.Inequality, cs.MX.sym('x', 2), np.array([-1, -2, -3]), np.array([3, 4]))


    def test_ctorDefaultBounds(self):
        '''Test Inequality ctor with default bounds
        '''
        i = ct.Inequality(cs.MX.sym('x', 3))
        nptest.assert_equal(i.lb, np.array(-cs.DM.inf(3)))
        nptest.assert_equal(i.ub, np.array(cs.DM.inf(3)))


    def test_ctorDefaultLowerBound(self):
        '''Test Inequality ctor with default lower bound
        '''
        i = ct.Inequality(cs.MX.sym('x', 3), ub=cs.DM([1, 2, 3]))
        nptest.assert_equal(i.lb, np.array(-cs.DM.inf(3)))
        nptest.assert_equal(i.ub, np.array(cs.DM([1, 2, 3])))


    def test_ctorDefaultUpperBound(self):
        '''Test Inequality ctor with default upper bound
        '''
        i = ct.Inequality(cs.MX.sym('x', 3), lb=cs.DM([1, 2, 3]))
        nptest.assert_equal(i.lb, np.array(cs.DM([1, 2, 3])))
        nptest.assert_equal(i.ub, np.array(cs.DM.inf(3)))


    def test_numel(self):
        '''Test Inequality numel() function
        '''
        i = ct.Inequality(cs.MX.sym('x', 3))
        self.assertEqual(i.numel(), 3)


class BoundedVariableTest(unittest.TestCase):
    """
    Tests for BoundedVariable class
    """

    def test_ctor(self):
        '''Test BoundedVariable ctor
        '''

        i = ct.BoundedVariable(name='x', lb=np.array([-1, -2]), ub=np.array([3, 4]), nominal=cs.DM([1, 2]))

        nptest.assert_equal(i.lb, np.array(cs.DM([-1, -2])))
        nptest.assert_equal(i.ub, np.array(cs.DM([3, 4])))
        nptest.assert_equal(i.nominal, np.array(cs.DM([1, 2])))


    def test_ctorDefaultBounds(self):
        '''Test BoundedVariable ctor with default bounds
        '''

        i = ct.BoundedVariable(name='x', nominal=cs.DM([1, 2]))

        nptest.assert_equal(i.lb, np.array(-cs.DM.inf(2)))
        nptest.assert_equal(i.ub, np.array(cs.DM.inf(2)))


    def test_numel(self):
        '''Test BoundedVariable numel() function
        '''
        i = ct.BoundedVariable('x', nominal=cs.DM.zeros(4))
        self.assertEqual(i.numel(), 4)


if __name__ == '__main__':
    unittest.main()