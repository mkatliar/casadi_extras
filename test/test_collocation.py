"""
Tests for the collocation module
"""


import unittest
import numpy.testing as nptest
import numpy as np
import casadi as cs
import casadi_tools as ct

import casadi_tools.collocation as cl
from casadi_tools import dae_model

import matplotlib.pyplot as plt


class PdqTest(unittest.TestCase):
    """
    Tests for the Pdq class
    """

    def test_interpolatingFunction(self):
        def f(x):
            return np.vstack((
                np.exp(x) * np.sin(5 * x),
                np.cos(10 * x) * np.exp(np.sin(10 * x))
            ))

            #return np.cos(10 * x) * np.exp(np.sin(10 * x))


        pdq = cl.Pdq(t=[-1, 1], poly_order=99)

        y = f(pdq.collocationPoints)
        xx = np.linspace(pdq.t0, pdq.tf, 400)
        yy = f(xx)

        fp = cl.polynomialInterpolator(pdq.collocationPoints)
        fb = cl.barycentricInterpolator(pdq.collocationPoints)
        yp = fp(y, xx)
        yb = fb(y, xx)
        
        '''
        M = y.shape[0]
        for i in range(M):
            plt.subplot(M, 3, 1 + 3 * i)
            plt.plot(xx, np.vstack((yy[i, :], yp[i, :], yb[i, :])).T)
            plt.plot(pdq.x, y[i, :].T, '.')
            plt.legend(['actual', 'polynomial interpolated', 'barycentric interpolated', 'cl-points'])
            plt.grid(True)

            plt.subplot(M, 3, 2 + 3 * i)
            plt.plot(xx, (yp[i, :] - yy[i, :]).T)
            plt.legend(['polynomial interpolation error'])
            plt.grid(True)

            plt.subplot(M, 3, 3 + 3 * i)
            plt.plot(xx, (yb[i, :] - yy[i, :]).T)
            plt.legend(['barycentric interpolation error'])
            plt.grid(True)

        plt.show()
        '''

        self.assertTrue(np.all(np.isfinite(yp)))
        self.assertTrue(np.all(np.isfinite(yb)))
        np.testing.assert_allclose(yb, yy, atol=1e-8)


class ChebTest(unittest.TestCase):
    """
    Test for the cheb function
    """

    def test_0(self):
        D, t = cl.cheb(0, t0=-1, tf=1)

        nptest.assert_allclose(t, np.array([1]))
        nptest.assert_allclose(D, np.array([[0]]))


    def test_1(self):
        D, t = cl.cheb(1, t0=-1, tf=1)

        nptest.assert_allclose(np.flip(t, 0), np.array([1, -1]))
        nptest.assert_allclose(np.rot90(D, 2), np.array([
            [0.5, -0.5],
            [0.5, -0.5]
        ]))


    def test_2(self):
        D, _ = cl.cheb(2, t0=-1, tf=1)

        nptest.assert_allclose(np.rot90(D, 2), np.array([
            [1.5, -2, 0.5],
            [0.5, 0, -0.5],
            [-0.5, 2, -1.5]
        ]))


    def test_3(self):
        D, _ = cl.cheb(3, t0=-1, tf=1)

        #nptest.assert_almost_equal(x, np.array([1, -1]))
        nptest.assert_allclose(np.rot90(D, 2), np.array([
            [3.1667, -4.0000, 1.3333, -0.5000],
            [1.0000, -0.3333, -1.0000, 0.3333],
            [-0.3333, 1.0000, 0.3333, -1.0000],
            [0.5000, -1.3333, 4.0000, -3.1667]]), atol=1e-4, rtol=0)


class DiffTest(unittest.TestCase):
    """
    Test differentiation matrices
    """
    def test_diff_cheb(self):
        """Test differentiation using Chebyshev matrix
        """
        N = 20
        D, t = cl.cheb(N, t0=0, tf=2)

        y = np.exp(t) * np.sin(5 * t)
        nptest.assert_allclose(np.dot(D, y), y + np.exp(t) * np.cos(5 * t) * 5, atol=1e-8, rtol=0)


class IvpTest(unittest.TestCase):
    """
    Solving initial value problem with collocation method test
    """

    def test_ivp(self):
        """Test solving IVP with collocation
        """
        x = cs.MX.sym('x')
        xdot = x

        N = 10
        tf = 1
        pdq = cl.Pdq(t=[0, tf], poly_order=N)

        var = ct.struct_symMX([
            ct.entry('X', shape=(1, N))
        ])

        f = cs.Function('f', [x], [xdot])
        F = f.map(N, 'serial')

        x0 = cs.MX.sym('x0')
        eq = cs.Function('eq', [var, x0], 
            [cs.reshape(F(cs.horzcat(x0, var['X', :, : -1])) - pdq.derivative(cs.horzcat(x0, var['X'])), var.size, 1)])
        rf = cs.rootfinder('rf', 'newton', eq)

        sol = var(rf(var(0), 1))
        nptest.assert_allclose(sol['X', :, -1], 1 * np.exp(1 * tf))


    def test_collocation_integrator_ode(self):
        """Test collocation_integrator function with ODE
        """
        x = cs.MX.sym('x')
        xdot = x

        N = 10
        tf = 1
        
        dae = dae_model.Dae(x=x, ode=xdot)
        pdq = cl.Pdq([0, tf], poly_order=N)
        
        integrator = cl.collocationIntegrator('integrator', dae, pdq)
        sol = integrator(x0=1)

        nptest.assert_allclose(sol['xf'], 1 * np.exp(1 * tf))


    def test_collocation_integrator_ode_time_dependent(self):
        """Test collocation_integrator function with time-dependent ODE
        """
        x = cs.MX.sym('x')
        t = cs.MX.sym('t')
        xdot = t ** 2

        N = 10
        x0 = 1.1
        t0 = 2.4
        tf = 3.9
        
        dae = dae_model.Dae(x=x, ode=xdot, t=t)
        pdq = cl.Pdq([t0, tf], poly_order=N)
        
        integrator = cl.collocationIntegrator('integrator', dae, pdq)
        sol = integrator(x0=x0)

        nptest.assert_allclose(sol['xf'], x0 + tf ** 3 / 3 - t0 ** 3 / 3)


    def test_collocation_integrator_dae(self):
        """Test collocation_integrator function with DAE
        """
        x = cs.MX.sym('x')
        z = cs.MX.sym('z')
        u = cs.MX.sym('u')
        xdot = z + u
        alg = z*z - x

        N = 10
        tf = 2
        
        dae = dae_model.Dae(x=x, z=z, u=u, ode=xdot, alg=alg)
        pdq = cl.Pdq([0, tf], poly_order=N)
        
        integrator = cl.collocationIntegrator('integrator', dae, pdq)

        # The analytic solution is
        # x(t) = 1/4 * (t + c)^2, c = 1 (+-) 2 * sqrt(x(-1))
        
        for x0, z0, sign in [(0.1, 1, 1), (1.1, -1, -1)]:
            sol = integrator(x0=x0, z0=z0, u=0)
            nptest.assert_allclose(sol['xf'], (tf**2 + sign * 4 * tf * np.sqrt(x0) + 4 * x0) / 4)


    def test_collocation_integrator_dae_with_quad(self):
        """Test collocation_integrator function with DAE and quadrature
        """
        x = cs.MX.sym('x')
        z = cs.MX.sym('z')
        xdot = z
        alg = z*z - x
        q = x

        N = 10
        tf = 2
        
        dae = dae_model.Dae(x=x, z=z, ode=xdot, alg=alg, quad=q)
        pdq = cl.Pdq([0, tf], poly_order=N)
        
        integrator = cl.collocationIntegrator('integrator', dae, pdq)

        # The analytic solution is
        # x(t) = 1/4 * (t + c)^2, c = 1 (+-) 2 * sqrt(x(-1))
        
        for x0, z0, sign in [(0.1, 1, 1), (1.1, -1, -1)]:
            sol = integrator(x0=x0, z0=z0)
            c = 1 + sign * 2 * np.sqrt(x0)
            nptest.assert_allclose(sol['xf'], (1 + c)**2 / 4)
            nptest.assert_allclose(sol['qf'], ((1 + c)**3 - (-1 + c)**3) / 12)
            #nptest.assert_allclose(sol['xf'], (tf**2 + sign * 4 * tf * np.sqrt(x0) + 4 * x0) / 4)


    def test_directCollocationSimple(self):
        """Test direct collocation on a very simple model
        """

        # Double integrator model
        x = ct.struct_symMX([
            ct.entry('q'),
            ct.entry('v')
        ])
        
        u = cs.MX.sym('u')
        
        ode = ct.struct_MX(x)
        ode['q'] = x['v']
        ode['v'] = u

        quad = x['v']**2

        NT = 2  # number of control intervals
        N = 3   # number of collocation intervals per control interval
        ts = 1  # time step

        # PDQ
        pdq = cl.Pdq(np.arange(NT + 1) * ts, poly_order=N)
        
        # DAE model
        dae = dae_model.Dae(x=x.cat, ode=ode.cat, u=u, quad=quad)

        # Create direct collocation scheme
        scheme = cl.CollocationScheme(dae, pdq)

        # Optimization variable
        w = scheme.combine(['x', 'z', 'u'])

        # Objective
        f = scheme.q[:, -1]

        # Constraints
        g = ct.struct_MX([
            ct.entry('eq', expr=scheme.eq),
            ct.entry('initial', expr=scheme.x[:, 0]),     # q0 = 0, v0 = 0
            ct.entry('final', expr=scheme.x[:, -1] - np.array([1, 0]))   # qf = 1, vf = 0
        ])

        # Make NLP
        nlp = {'x': w, 'g': g, 'f': f}

        # Init NLP solver
        opts = {'ipopt.linear_solver': 'ma86'}
        solver = cs.nlpsol('solver', 'ipopt', nlp, opts)

        # Run NLP solver
        sol = solver(lbg=0, ubg=0)
        sol_w = w(sol['x'])

        # Check agains the known solution
        nptest.assert_allclose(sol_w['u'], [[1, -1]])
        nptest.assert_allclose(sol['f'], 2. / 3.)


    def test_directCollocationReach(self):
        """Test direct collocation on a toy problem

        The problem: bring the double integrator from state [0, 0] to state [1, 0]
        while minimizing L2 norm of the control input.
        """

        # Double integrator model
        x = ct.struct_symMX([
            ct.entry('q'),
            ct.entry('v')
        ])
        
        u = cs.MX.sym('u')
        
        ode = ct.struct_MX(x)
        ode['q'] = x['v']
        ode['v'] = u

        quad = u**2

        NT = 5  # number of control intervals
        N = 3   # number of collocation intervals per control interval
        ts = 1  # time step

        # PDQ
        pdq = cl.Pdq(np.arange(NT + 1) * ts, poly_order=N)
        
        # DAE model
        dae = dae_model.Dae(x=x.cat, ode=ode.cat, u=u, quad=quad)

        # Create direct collocation scheme
        scheme = cl.CollocationScheme(dae, pdq)

        # Optimization variable
        w = scheme.combine(['x', 'z', 'u'])

        # Objective
        f = scheme.q[:, -1]

        # Constraints
        g = ct.struct_MX([
            ct.entry('eq', expr=scheme.eq),
            ct.entry('initial', expr=scheme.x0[:, 0]),     # q0 = 0, v0 = 0
            ct.entry('final', expr=scheme.x[:, -1] - np.array([1, 0]))   # qf = 1, vf = 0
        ])

        # Make NLP
        nlp = {'x': w, 'g': g, 'f': f}

        # Init NLP solver
        opts = {'ipopt.linear_solver': 'ma86'}
        solver = cs.nlpsol('solver', 'ipopt', nlp, opts)

        # Run NLP solver
        sol = solver(lbg=0, ubg=0)
        sol_w = w(sol['x'])

        # Check against the known solution
        nptest.assert_allclose(sol_w['u'], [[0.2, 0.1, 0, -0.1, -0.2]], atol=1e-16)

        plt.plot(scheme.t, sol_w['x'].T, '.-')
        plt.hold(True)

        fi = pdq.interpolator()
        t = np.linspace(0, NT * ts, num=50)
        plt.plot(t, fi(sol_w['x'], t).T)

        plt.show()


if __name__ == '__main__':
    #import sys;sys.argv = ['', 'IvpTest.test_collocation_integrator_dae_with_quad']
    unittest.main()