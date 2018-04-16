"""
Tests for the struct module
"""


import unittest
import numpy.testing as nptest
import numpy as np
import casadi as cs
import casadi_tools as ct


class StructTest(unittest.TestCase):
    """
    Tests for structs
    """

    def test_pythonStructToCasadiStruct(self):
        '''Test creation of CasADi structs from Python structs.
        '''

        s = ct.struct_symMX([
            ct.entry('x', shape=2),
            ct.entry('y', shape=3)
        ])

        sn = s({'x': np.array([2, 3]), 'y': 42})

        nptest.assert_equal(sn['x'], np.array(cs.DM([2, 3])))
        nptest.assert_equal(sn['y'], np.array(cs.DM([42, 42, 42])))


    def test_pythonStructToCasadiStructInvalidKeys(self):
        '''Test creation of CasADi structs from Python structs with invalid keys.
        '''

        s = ct.struct_symMX([
            ct.entry('x', shape=2),
            ct.entry('y', shape=3)
        ])

        self.assertRaises(Exception, s, {'x': np.array([2, 3]), 'yy': 42})


    def test_pythonStructToCasadiStructInvalidShape(self):
        '''Test creation of CasADi structs from Python structs with invalid shape.
        '''

        s = ct.struct_symMX([
            ct.entry('x', shape=2),
            ct.entry('y', shape=3)
        ])

        self.assertRaises(Exception, s, {'x': np.array([2, 3]), 'y': np.array([42, 43])})


if __name__ == '__main__':
    #import sys;sys.argv = ['', 'IvpTest.test_collocation_integrator_dae_with_quad']
    unittest.main()