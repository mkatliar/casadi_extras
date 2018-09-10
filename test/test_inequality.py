"""
Tests for the inequality module
"""


import unittest
import numpy.testing as nptest
import numpy as np
import casadi as cs
import casadi_extras as ce


class InequalityTest(unittest.TestCase):
    """
    Tests for Inequality class
    """

    def test_ctor(self):
        '''Test Inequality ctor
        '''

        i = ce.Inequality(expr=cs.MX.sym('x', 2), lb=np.array([-1, -2]), ub=np.array([3, 4]))

        nptest.assert_equal(i.lb, np.array(cs.DM([-1, -2])))
        nptest.assert_equal(i.ub, np.array(cs.DM([3, 4])))


    def test_ctorInconsistentShape(self):
        '''Test Inequality ctor with arguments of inconsistent shape
        '''

        self.assertRaises(Exception, ce.Inequality, cs.MX.sym('x', 2), np.array([-1, -2, -3]), np.array([3, 4]))


    def test_ctorDefaultBounds(self):
        '''Test Inequality ctor with default bounds
        '''
        i = ce.Inequality(cs.MX.sym('x', 3))
        nptest.assert_equal(i.lb, np.array(-cs.DM.inf(3)))
        nptest.assert_equal(i.ub, np.array(cs.DM.inf(3)))


    def test_ctorDefaultLowerBound(self):
        '''Test Inequality ctor with default lower bound
        '''
        i = ce.Inequality(cs.MX.sym('x', 3), ub=cs.DM([1, 2, 3]))
        nptest.assert_equal(i.lb, np.array(-cs.DM.inf(3)))
        nptest.assert_equal(i.ub, np.array(cs.DM([1, 2, 3])))


    def test_ctorDefaultUpperBound(self):
        '''Test Inequality ctor with default upper bound
        '''
        i = ce.Inequality(cs.MX.sym('x', 3), lb=cs.DM([1, 2, 3]))
        nptest.assert_equal(i.lb, np.array(cs.DM([1, 2, 3])))
        nptest.assert_equal(i.ub, np.array(cs.DM.inf(3)))


    def test_numel(self):
        '''Test Inequality numel() function
        '''
        i = ce.Inequality(cs.MX.sym('x', 3))
        self.assertEqual(i.numel(), 3)


    def test_empty(self):
        '''Test empty Inequality
        '''
        i = ce.Inequality()
        self.assertEqual(i.numel(), 0)


    def test_vertcat_list(self):
        '''Test Inequality vertical concatenation as a list
        '''
        b1 = np.random.rand(3, 2)
        n1 = np.random.rand(3)
        i1 = ce.Inequality(cs.MX.sym('x', 3), lb=np.min(b1, axis=1), ub=np.max(b1, axis=1), nominal=n1)

        b2 = np.random.rand(4, 2)
        n2 = np.random.rand(4)
        i2 = ce.Inequality(cs.MX.sym('y', 4), lb=np.min(b2, axis=1), ub=np.max(b2, axis=1), nominal=n2)

        i3 = ce.inequality.vertcat([i1, i2])

        self.assertEqual(i3.numel(), 7)
        nptest.assert_equal(np.array(i3.lb), np.array(cs.vertcat(i1.lb, i2.lb)))
        nptest.assert_equal(np.array(i3.ub), np.array(cs.vertcat(i1.ub, i2.ub)))
        nptest.assert_equal(np.array(i3.nominal), np.array(cs.vertcat(i1.nominal, i2.nominal)))


    def test_vertcat_dict(self):
        '''Test Inequality vertical concatenation as a dict
        '''
        b1 = np.random.rand(3, 2)
        n1 = np.random.rand(3)
        i1 = ce.Inequality(cs.MX.sym('x', 3), lb=np.min(b1, axis=1), ub=np.max(b1, axis=1), nominal=n1)

        b2 = np.random.rand(4, 2)
        n2 = np.random.rand(4)
        i2 = ce.Inequality(cs.MX.sym('y', 4), lb=np.min(b2, axis=1), ub=np.max(b2, axis=1), nominal=n2)

        i3 = ce.inequality.vertcat({'i1': i1, 'i2': i2})

        self.assertEqual(i3.numel(), 7)
        nptest.assert_equal(np.array(i3.lb.cat), np.array(cs.vertcat(i1.lb, i2.lb)))
        nptest.assert_equal(np.array(i3.ub.cat), np.array(cs.vertcat(i1.ub, i2.ub)))
        nptest.assert_equal(np.array(i3.nominal.cat), np.array(cs.vertcat(i1.nominal, i2.nominal)))
        
        nptest.assert_equal(np.array(i3.lb['i1']), np.array(i1.lb))
        nptest.assert_equal(np.array(i3.ub['i1']), np.array(i1.ub))
        nptest.assert_equal(np.array(i3.lb['i2']), np.array(i2.lb))
        nptest.assert_equal(np.array(i3.ub['i2']), np.array(i2.ub))
        nptest.assert_equal(np.array(i3.nominal['i2']), np.array(i2.nominal))


class BoundedVariableTest(unittest.TestCase):
    """
    Tests for BoundedVariable class
    """

    def test_ctor(self):
        '''Test BoundedVariable ctor
        '''

        i = ce.BoundedVariable(name='x', lb=np.array([-1, -2]), ub=np.array([3, 4]), nominal=cs.DM([1, 2]))

        nptest.assert_equal(i.lb, np.array(cs.DM([-1, -2])))
        nptest.assert_equal(i.ub, np.array(cs.DM([3, 4])))
        nptest.assert_equal(i.nominal, np.array(cs.DM([1, 2])))


    def test_ctorDefaultBounds(self):
        '''Test BoundedVariable ctor with default bounds
        '''

        i = ce.BoundedVariable(name='x', nominal=cs.DM([1, 2]))

        nptest.assert_equal(i.lb, np.array(-cs.DM.inf(2)))
        nptest.assert_equal(i.ub, np.array(cs.DM.inf(2)))


    def test_numel(self):
        '''Test BoundedVariable numel() function
        '''
        i = ce.BoundedVariable('x', nominal=cs.DM.zeros(4))
        self.assertEqual(i.numel(), 4)


if __name__ == '__main__':
    unittest.main()