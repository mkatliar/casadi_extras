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
Tests for the struct module
"""


import unittest
import numpy.testing as nptest
import numpy as np
import casadi as cs
import casadi_extras as ce


class StructTest(unittest.TestCase):
    """
    Tests for structs
    """

    def test_pythonStructToCasadiStruct(self):
        '''Test creation of CasADi structs from Python structs.
        '''

        s = ce.struct_symMX([
            ce.entry('x', shape=2),
            ce.entry('y', shape=3)
        ])

        sn = s({'x': np.array([2, 3]), 'y': 42})

        nptest.assert_equal(sn['x'], np.array(cs.DM([2, 3])))
        nptest.assert_equal(sn['y'], np.array(cs.DM([42, 42, 42])))


    def test_pythonStructToCasadiStructInvalidKeys(self):
        '''Test creation of CasADi structs from Python structs with invalid keys.
        '''

        s = ce.struct_symMX([
            ce.entry('x', shape=2),
            ce.entry('y', shape=3)
        ])

        self.assertRaises(Exception, s, {'x': np.array([2, 3]), 'yy': 42})


    def test_pythonStructToCasadiStructInvalidShape(self):
        '''Test creation of CasADi structs from Python structs with invalid shape.
        '''

        s = ce.struct_symMX([
            ce.entry('x', shape=2),
            ce.entry('y', shape=3)
        ])

        self.assertRaises(Exception, s, {'x': np.array([2, 3]), 'y': np.array([42, 43])})


    def test_parseMatrix(self):
        '''Test parsing a matrix as CasADi structures.
        '''

        s = ce.struct_symMX([
            ce.entry('x', shape=2),
            ce.entry('y', shape=3)
        ])

        data = np.array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
            [10, 11, 12],
            [13, 14, 15]
        ])

        res = s.parseMatrix(data)
        
        self.assertEqual(set(res), {'x', 'y'})

        nptest.assert_equal(np.array(res['x']),
            cs.DM([
                [1, 2, 3],
                [4, 5, 6]
            ])
        )
            
        nptest.assert_equal(np.array(res['y']),
            cs.DM([
                [7, 8, 9],
                [10, 11, 12],
                [13, 14, 15]
            ])
        )


if __name__ == '__main__':
    #import sys;sys.argv = ['', 'IvpTest.test_collocation_integrator_dae_with_quad']
    unittest.main()