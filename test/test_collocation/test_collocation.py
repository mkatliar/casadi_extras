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


class PolynomialBasisTest(unittest.TestCase):
    '''
    Tests for PolynomialBasis class
    '''

    def test_interpolationMatrix(self):
        N = 3
        tau = cl.collocationPoints(N, 'legendre')
        #tau = np.array([-1, 0])
        basis = cl.PolynomialBasis(tau)

        # Coefficients of the collocation equation
        C = np.zeros((N+1,N+1))

        # Coefficients of the continuity equation
        D = np.zeros(N+1)

        # Coefficients of the quadrature function
        B = np.zeros(N+1)

        # Construct polynomial basis manually
        for j in range(N + 1):
            # Construct Lagrange polynomials to get the polynomial basis at the collocation point
            p = np.poly1d([1])
            for r in range(N + 1):
                if r != j:
                    p *= np.poly1d([1, -tau[r]]) / (tau[j]-tau[r])

            # Evaluate the polynomial at the final time to get the coefficients of the continuity equation
            D[j] = p(1.0)

            # Evaluate the time derivative of the polynomial at all collocation points to get the coefficients of the continuity equation
            pder = np.polyder(p)
            for r in range(N + 1):
                C[j,r] = pder(tau[r])

            # Evaluate the integral of the polynomial to get the coefficients of the quadrature function
            pint = np.polyint(p)
            B[j] = pint(1.0)

        nptest.assert_allclose(basis.D, C.T)
        nptest.assert_allclose(basis.interpolationMatrix(1.0), np.atleast_2d(D))
        nptest.assert_allclose(basis.interpolationMatrix(0), np.atleast_2d((np.arange(N + 1) == 0).astype(float)))


    def test_interpolationMatrixChebyshev(self):

        N = 3
        tau = cl.collocationPoints(N, 'chebyshev')
        basis = cl.PolynomialBasis(tau)

        nptest.assert_allclose(basis.interpolationMatrix([0.0, 1.0]), np.array([
            [1, 0, 0, 0],
            [0, 0, 0, 1]
        ]))


@unittest.skip('Pdq class test disable, Pdq class is frozen')
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


    def test_expandInput(self):
        pdq = cl.Pdq(t=[0, 1, 2], poly_order=2)

        u = cs.DM([
            [1, 2],
            [3, 4]
        ])

        nptest.assert_equal(np.array(pdq.expandInput(u)),
            cs.DM([
                [1, 1, 1, 2, 2, 2],
                [3, 3, 3, 4, 4, 4]
            ])
        )


    def test_ivp(self):
        """Test solving IVP with collocation
        """
        x = cs.MX.sym('x')
        xdot = x

        N = 10
        tf = 1
        pdq = cl.Pdq(t=[0, tf], poly_order=N)

        X = cs.MX.sym('X', 1, N)
        f = cs.Function('f', [x], [xdot])
        F = f.map(N, 'serial')

        x0 = cs.MX.sym('x0')
        eq = cs.Function('eq', [cs.vec(X), x0], 
            [cs.vec(F(X) - pdq.derivative(cs.horzcat(x0, X))[:, 1 :])])
        rf = cs.rootfinder('rf', 'newton', eq)

        sol = cs.reshape(rf(cs.DM.zeros(X.shape), 1), X.shape)
        nptest.assert_allclose(sol[:, -1], 1 * np.exp(1 * tf))


    def test_ivp1(self):
        """Test solving IVP1 with collocation
        """
        x = cs.MX.sym('x')
        xdot = x

        N = 10
        tf = 1
        pdq = cl.Pdq(t=[0, tf], poly_order=N)

        X = cs.MX.sym('X', 1, N + 1)
        f = cs.Function('f', [x], [xdot])
        F = f.map(N + 1, 'serial')

        x0 = cs.MX.sym('x0')
        eq = cs.Function('eq', [cs.vec(X)], 
            [cs.vec(F(X) - pdq.derivative(X))])
        rf = cs.rootfinder('rf', 'newton', eq)

        sol = cs.reshape(rf(cs.DM.zeros(X.shape)), X.shape)
        nptest.assert_allclose(sol[:, -1], 1 * np.exp(1 * tf))


class ChebTest(unittest.TestCase):
    """
    Test for the cheb function
    """

    def test_0(self):
        D, t = cl.cheb(0, t0=1, tf=-1)

        nptest.assert_allclose(t, np.array([1]))
        nptest.assert_allclose(D, np.array([[0]]))


    def test_1(self):
        D, t = cl.cheb(1, t0=1, tf=-1)

        nptest.assert_allclose(t, np.array([1, -1]))
        nptest.assert_allclose(D, np.array([
            [0.5, -0.5],
            [0.5, -0.5]
        ]))


    def test_2(self):
        D, _ = cl.cheb(2, t0=1, tf=-1)

        nptest.assert_allclose(D, np.array([
            [1.5, -2, 0.5],
            [0.5, 0, -0.5],
            [-0.5, 2, -1.5]
        ]), atol=1e-10)


    def test_3(self):
        D, _ = cl.cheb(3, t0=1, tf=-1)

        #nptest.assert_almost_equal(x, np.array([1, -1]))
        nptest.assert_allclose(D, np.array([
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


class CollocationIntegratorTest(unittest.TestCase):
    '''Unit tests for collocation integrator.
    '''

    def test_collocationIntegrator_ode(self):
        """Test collocationIntegrator function with ODE
        """
        x = cs.MX.sym('x')
        xdot = x

        N = 9
        tf = 1
        
        dae = dae_model.SemiExplicitDae(x=x, ode=xdot)
        integrator = cl.collocationIntegrator('integrator', dae, t=[0, tf], order=N)
        sol = integrator(x0=1)

        nptest.assert_allclose(sol['xf'], 1 * np.exp(1 * tf))


    def test_collocationIntegrator_ode_with_input(self):
        """Test collocation_integrator function with ODE and control input
        """
        x = cs.MX.sym('x')
        u = cs.MX.sym('u')
        xdot = x + u

        N = 9
        tf = 1
        
        dae = dae_model.SemiExplicitDae(x=x, u=u, ode=xdot)
        integrator = cl.collocationIntegrator('integrator', dae, t=[0, tf], order=N)

        x0 = 1
        u0 = 0.1
        sol = integrator(x0=1, u=u0)

        nptest.assert_allclose(sol['xf'], (x0 + u0) * np.exp(tf) - u0)


    def test_collocationIntegrator_ode_with_input_multistep(self):
        """Test collocation_integrator function with ODE and multi-step control input
        """
        x = cs.MX.sym('x')
        u = cs.MX.sym('u')
        xdot = x + u

        N = 9
        t = [0, 1, 2]
        
        dae = dae_model.SemiExplicitDae(x=x, u=u, ode=xdot)
        integrator = cl.collocationIntegrator('integrator', dae, t=t, order=N)

        x0 = 1
        u = np.atleast_2d([[0.1, -0.2]])
        sol = integrator(x0=1, u=u)

        xf = sol['xf']
        nptest.assert_allclose(xf[:, 0], np.atleast_2d((x0 + u[:, 0]) * np.exp(t[1] - t[0]) - u[:, 0]))
        nptest.assert_allclose(xf[:, 1], np.atleast_2d((xf[:, 0] + u[:, 1]) * np.exp(t[2] - t[1]) - u[:, 1]))


    def test_collocation_integrator_ode_time_dependent(self):
        """Test collocation_integrator function with time-dependent ODE
        """
        x = cs.MX.sym('x')
        t = cs.MX.sym('t')
        xdot = t ** 2

        N = 9
        x0 = 1.1
        t0 = 2.4
        tf = 3.9
        
        dae = dae_model.SemiExplicitDae(x=x, ode=xdot, t=t)        
        integrator = cl.collocationIntegrator('integrator', dae, t=[t0, tf], order=N)
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

        N = 9
        tf = 2
        
        dae = dae_model.SemiExplicitDae(x=x, z=z, u=u, ode=xdot, alg=alg)
        integrator = cl.collocationIntegrator('integrator', dae, t=[0, tf], order=N)

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

        N = 9
        tf = 2
        
        dae = dae_model.SemiExplicitDae(x=x, z=z, ode=xdot, alg=alg, quad=q)
        integrator = cl.collocationIntegrator('integrator', dae, t=[0, tf], order=N)

        # The analytic solution is
        # x(t) = 1/4 * (t + c)^2, c = 1 (+-) 2 * sqrt(x(-1))
        
        for x0, z0, sign in [(0.1, 1, 1), (1.1, -1, -1)]:
            sol = integrator(x0=x0, z0=z0)
            c = 1 + sign * 2 * np.sqrt(x0)
            nptest.assert_allclose(sol['xf'], (1 + c)**2 / 4)
            nptest.assert_allclose(sol['qf'], ((1 + c)**3 - (-1 + c)**3) / 12)
            #nptest.assert_allclose(sol['xf'], (tf**2 + sign * 4 * tf * np.sqrt(x0) + 4 * x0) / 4)


class CollocationSchemeTest(unittest.TestCase):
    '''Tests for CollocationScheme.
    '''

    def test_ivp(self):
        """Test solving IVP with collocation
        """
        x = cs.MX.sym('x')
        xdot = x
        dae = dae_model.SemiExplicitDae(x=x, ode=xdot)

        N = 4
        tf = 1
        scheme = cl.CollocationScheme(dae=dae, t=[0, tf], order=N, method='legendre')

        x0 = cs.MX.sym('x0')
        var = scheme.combine(['x', 'K'])

        eqf = cs.Function('eq', [cs.vec(var), x0], [cs.vertcat(scheme.eq, scheme.x[:, 0] - x0)])
        rf = cs.rootfinder('rf', 'newton', eqf)

        sol = var(rf(var(0), 1))
        nptest.assert_allclose(sol['x', :, -1], np.atleast_2d(1 * np.exp(1 * tf)))


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
        N = 3   # number of collocation points per interval
        ts = 1  # time step

        # DAE model
        dae = dae_model.SemiExplicitDae(x=x.cat, ode=ode.cat, u=u, quad=quad)

        # Create direct collocation scheme
        scheme = cl.CollocationScheme(dae=dae, t=np.arange(NT + 1) * ts, order=N)

        # Optimization variable
        w = scheme.combine(['x', 'K', 'Z', 'u'])

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
        N = 3   # number of collocation points per interval
        ts = 1  # time step

        # DAE model
        dae = dae_model.SemiExplicitDae(x=x.cat, ode=ode.cat, u=u, quad=quad)

        # Create direct collocation scheme
        scheme = cl.CollocationScheme(dae=dae, t=np.arange(NT + 1) * ts, order=N)

        # Optimization variable
        w = scheme.combine(['x', 'K', 'u'])

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

        # Check against the known solution
        nptest.assert_allclose(sol_w['u'], [[0.2, 0.1, 0, -0.1, -0.2]], atol=1e-16)

        plt.plot(scheme.t, sol_w['x'].T, 'o')
        plt.hold(True)

        plt.plot(scheme.tc, scheme.evalX(sol_w['x'], sol_w['K']).T, 'x')

        fi = scheme.piecewisePolyX(sol_w['x'], sol_w['K'])
        t, val = fi.discretize(ts / 10.)
        plt.plot(t, val.T)

        plt.grid(True)
        plt.show()


"""
class CollocationSimulatorTest(unittest.TestCase):
    '''Tests for CollocationSimulator.
    '''

    def test_simulate(self):
        '''Check if CollocationSimulator.simulate() works.
        '''

        x = cs.MX.sym('x')
        u = cs.MX.sym('u')
        xdot = x * u

        dae = dae_model.SemiExplicitDae(x=x, u=u, ode=xdot)        
        simulator = cl.CollocationSimulator(dae, t=[0, 1, 2], poly_order=5)

        def input(t):
            return 1 if t < 1 else -1
        
        sol = simulator.simulate(x0=1, input=input)

        #nptest.assert_allclose(sol['xf'], 1 * np.exp(1 * tf))
"""


if __name__ == '__main__':
    #import sys;sys.argv = ['', 'IvpTest.test_collocation_integrator_dae_with_quad']
    unittest.main()