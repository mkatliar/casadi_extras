"""
Collocation method
"""
import numpy as np
import casadi as cs
import casadi_tools as ct

from .polynomial import PolynomialBasis, collocationPoints, barycentricInterpolator
from .butcher import butcherTableuForCollocationMethod
from .piecewise_poly import PiecewisePoly


class Pdq(object):
    """Polynomial-based Differential Quadrature (PDQ).

    See "Differential Quadrature and Its Application in Engineering: 
    Engineering Applications, Chang Shu, Springer, 2000, ISBN 978-1-85233-209-9"

    https://link.springer.com/content/pdf/10.1007%2F978-1-4471-0407-0.pdf
    """

    def __init__(self, t, poly_order=5, polynomial_type='chebyshev'):
        """Constructor

        @param t N+1 points defining N collocation intervals in ascending order.
        @param poly_order order of collocation polynomial.
        @param polynomial_type collocation polynomial type.
        """

        N = len(t) - 1

        # Make collocation points vector and differential matrices list
        collocation_points = []
        collocation_groups = []
        basis = []
        k = 0

        for i in range(N):
            t_i = collocationPoints(poly_order, polynomial_type) * (t[i + 1] - t[i]) + t[i]
            basis_i = PolynomialBasis(t_i)
            D_i = basis_i.D
            
            assert D_i.shape[0] == poly_order + 1 and D_i.shape[1] == poly_order + 1
            assert t_i.shape == (poly_order + 1,)

            # Ensure that the points are from left to right.
            assert np.all(np.diff(t_i) > 0)

            basis.append(basis_i)
            collocation_points.append(t_i[: -1])
            collocation_groups.append(np.arange(k, k + poly_order + 1))
            k += poly_order

        # Append the last point.
        # TODO: is it needed or not?
        collocation_points.append(t_i[-1])

        # Stack collocation points in one vector
        collocation_points = np.hstack(collocation_points)

        self._basis = basis
        self._collocationPoints = collocation_points

        # Indices of collocation points belonging to the same interval, including both ends.
        self._collocationGroups = collocation_groups
        self._intervalBounds = t
        self._polyOrder = poly_order

        # Big sparse differentiation matrix
        N = len(collocation_points)
        bigD = cs.DM(N, N)

        i = 0
        for b in basis:
            bigD[i : i + b.numPoints - 1, i : i + b.numPoints] = b.D[: -1, :]
            i += b.numPoints - 1

        bigD[-1, -basis[-1].numPoints :] = basis[-1].D[-1, :]

        self._bigD = bigD


    @property
    def collocationPoints(self):
        """Collocation points"""
        return self._collocationPoints


    @property
    def intervalBounds(self):
        """Interval bounds"""
        return self._intervalBounds


    @property
    def numIntervals(self):
        '''Number of collocation intervals'''
        return len(self._intervalBounds) - 1


    @property
    def polyOrder(self):
        """Degree of interpolating polynomial"""
        return self._polyOrder


    @property
    def t0(self):
        """The leftmost collocation point"""
        return self._collocationPoints[0]


    @property
    def tf(self):
        """The rightmost collocation point"""
        return self._collocationPoints[-1]


    def intervalLength(self):
        """Distance between the leftmost and the rightmost collocation points

        TODO: deprecate?
        """
        return self._collocationPoints[-1] - self._collocationPoints[0]


    def derivative(self, y):
        """Calculate derivative from function values
        """

        return cs.mtimes(y, self._bigD[: -1, :].T)


    def integral(self, dy):
        """Calculate integral from derivative values
        """

        y0 = cs.MX.zeros(dy.shape[0])
        y = []
        k = 0
        for b in self._basis:
            N = b.numPoints - 1

            # Calculate y the equation 
            # dy = cs.mtimes(y[:, 1 :], D[: -1, 1 :].T)
            #invD = cs.inv(D[: -1, 1 :])
            #y.append(cs.transpose(cs.mtimes(invD, cs.transpose(dy[:, k : k + N]))))
            y.append(cs.transpose(cs.solve(b.D[: -1, 1 :], cs.transpose(dy[:, k : k + N]))) + cs.repmat(y0, 1, N))
            k += N
            y0 = y[-1][:, -1]

        return cs.horzcat(*y)


    def expandInput(self, u):
        '''Return input at collocation points given the input u on collocation intervals.
        '''

        n = self.numIntervals
        assert u.shape[1] == n
        return cs.horzcat(*[cs.repmat(u[:, k], 1, len(self._collocationGroups[k]) - 1) for k in range(n)])


    def interpolator(self, continuity='both'):
        """Create interpolating function based on values at collocation points

        @param specifies continuity of the interpolated function at the interval boundaries:
        - 'left' means that the function in continuous from the left,
        - 'right' means that the function in continuous from the right,
        - 'both' means that the function is continuous both from the left and from the right.
        """

        return _PdqInterpolator(self, continuity)


class _PdqInterpolator(object):

    def __init__(self, pdq, continuity='both'):
        """Create interpolating function based on values at collocation points

        @param specifies continuity of the interpolated function at the interval boundaries:
        - 'left' means that the function in continuous from the left,
        - 'right' means that the function in continuous from the right,
        - 'both' means that the function is continuous both from the left and from the right.
        """

        # Transform collocation groups depending on the continuity option.
        groups = []

        for g in pdq._collocationGroups:
            if continuity == 'both':
                groups.append(g)
            elif continuity == 'left':
                groups.append(g[1 : ])
            elif continuity == 'right':
                groups.append(g[: -1])
            else:
                raise ValueError('Invalid "continuity" value {0} in Pdq.interpolator()'.format(continuity))

        self._basis = [PolynomialBasis(pdq._collocationPoints[g]) for g in groups]
        self._pdq = pdq
        self._continuity = continuity
        self._groups = groups
        

    def __call__(self, x, t):
        pdq = self._pdq
        continuity = self._continuity
        groups = self._groups

        expected_x_cols = pdq._collocationPoints.size if continuity == 'both' else pdq._collocationPoints.size - 1
        if x.shape[1] != expected_x_cols:
            raise ValueError('Invalid number of columns in interpolation point matrix')

        l = []
        
        for ti in np.atleast_1d(t):
            i = np.clip(np.searchsorted(pdq._intervalBounds, ti, 'right') - 1, 0, len(pdq._intervalBounds) - 2)  # interval index
            l.append(np.dot(x[:, groups[i]], self._basis[i].interpolationMatrix(ti).T))

        return np.hstack(l)


class CollocationScheme(object):
    """Collocation equations on multiple intervals
    for a given DAE model and differentiation matrix.
    """

    def __init__(self, dae, t, order, method='legendre', 
        parallelization='serial', tdp_fun=None, expand=True, repeat_param=False):

        """Constructor

        @param t time vector of length N+1 defining N collocation intervals
        @param order number of collocation points per interval
        @param method collocation method ('legendre', 'radau')
        @param dae DAE model
        @param parallelization parallelization of the outer map. Possible set of values is the same as for casadi.Function.map().

        @return Returns a dictionary with the following keys:
        'X' -- state at collocation points
        'Z' -- alg. state at collocation points
        'x0' -- initial state
        'eq' -- the expression eq == 0 defines the collocation equation. eq depends on X, Z, x0, p.
        'Q' -- quadrature values at collocation points depending on x0, X, Z, p.
        """

        # Convert whatever DAE to implicit DAE
        dae = dae.makeImplicit()

        M = order
        N = len(t) - 1

        #
        # Define variables and functions corresponfing to all control intervals
        # 

        K = cs.MX.sym('K', dae.nx, N * M)   # State derivatives at collocation points
        Z = cs.MX.sym('Z', dae.nz, N * M)   # Alg state at collocation points
        x = cs.MX.sym('x', dae.nx, N + 1)   # State at the ends of collocation intervals (t)

        u = cs.MX.sym('u', dae.nu, N)    # Input on collocation intervals
        U = cs.horzcat(*[cs.repmat(u[:, n], 1, M) for n in range(N)]) # Input at collocation points

        # Butcher tableau for the selected method
        butcher = butcherTableuForCollocationMethod(order, method)

        # Interval lengths
        h = np.diff(t)

        # Integrated state at collocation points
        X = cs.horzcat(
            *[cs.repmat(x[:, n], 1, M) + h[n] * cs.mtimes(K[:, n * M : (n + 1) * M], butcher.A.T) for n in range(N)])

        # Integrated state at the ends of collocation intervals
        xf = x[:, : -1] + cs.horzcat(*[h[n] * cs.mtimes(K[:, n * M : (n + 1) * M], butcher.b) for n in range(N)])
        
        # Points in time at which the collocation equations are calculated
        tc = np.hstack([t[n] + h[n] * butcher.c for n in range(N)])

        # Values of the time-dependent parameter
        if tdp_fun is not None:
            tdp_val = cs.horzcat(*[tdp_fun(t) for t in tc])
        else:
            assert dae.ntdp == 0
            tdp_val = np.zeros((0, tc.size))

        # DAE function
        dae_fun = dae.createFunction('dae', ['xdot', 'x', 'z', 'u', 'p', 't', 'tdp'], ['dae', 'quad'])
        if expand:
            dae_fun = dae_fun.expand()  # expand() for speed

        if repeat_param:
            reduce_in = []
            p = cs.MX.sym('P', dae.np, N * M)
        else:
            reduce_in = [4]
            p = cs.MX.sym('P', dae.np)

        dae_map = dae_fun.map('dae_map', parallelization, N * M, reduce_in, [])
        dae_out = dae_map(xdot=K, x=X, z=Z, u=U, p=p, t=tc, tdp=tdp_val)

        eqc = ct.struct_MX([
            ct.entry('collocation', expr=dae_out['dae']),
            ct.entry('continuity', expr=xf - x[:, 1 :]),
            ct.entry('param', expr=cs.diff(p, 1, 1))
        ])

        # Integrate the quadrature state
        quad = dae_out['quad']

        Q = []  # Integrated quadrature at collocation points
        q = [cs.MX.zeros(dae.nq)]  # Integrated quadrature at interval ends
        
        for n in range(N):
            Q.append(cs.repmat(q[-1], 1, M) + h[n] * cs.mtimes(quad[:, n * M : (n + 1) * M], butcher.A.T))
            q.append(q[-1] + h[n] * cs.mtimes(quad[:, n * M : (n + 1) * M], butcher.b))

        self._N = N
        self._M = M

        self._eq = eqc
        self._x = x
        self._X = X
        self._K = K
        self._Z = Z
        self._U = U
        self._u = u
        self._quad = quad
        self._Q = cs.horzcat(*Q)
        self._q = cs.horzcat(*q)
        self._p = p
        self._tc = tc
        self._butcher = butcher
        self._tdp = tdp_val
        self._t = t


    @property
    def pdq(self):
        """PDQ used by the collocation scheme"""
        return self._pdq


    @property
    def t(self):
        """Collocation intervals as time vector"""
        return self._t


    @property
    def tc(self):
        """Collocation points as time vector"""
        return self._tc


    @property
    def numTotalCollocationPoints(self):
        """Total number of collocation points"""
        return self._M * self._N


    @property
    def x(self):
        """State at interval ends"""
        return self._x


    @property
    def X(self):
        """State at collocation points"""
        return self._X


    def evalX(self, x, K):
        [X] = cs.substitute([self._X], [self._x, self._K], [x, K])
        return cs.evalf(X)


    @property
    def xdot(self):
        """State derivative at collocation points
        
        TODO: deprecate; the new name is K
        """
        return self._K


    @property
    def K(self):
        """State derivative at collocation points"""
        return self._K


    @property
    def Z(self):
        """Algebraic state at collocation points"""
        return self._Z


    @property
    def U(self):
        """Control input at collocation points"""
        return self._U


    @property
    def u(self):
        """Control input on intervals"""
        return self._u


    @property
    def p(self):
        """DAE model parameters"""
        return self._p


    @property
    def q(self):
        """Quadrature state at interval ends"""
        return self._q

    
    @property
    def Q(self):
        """Quadrature state at collocation points"""
        return self._Q


    @property
    def quad(self):
        """Quadrature state derivative at collocation points"""
        return self._quad
        

    '''
    @property
    def x0(self):
        """State at the beginning of each control interval
        
        TODO: deprecate?
        """
        return self._x0
    '''
        

    @property
    def eq(self):
        """Right-hand side of collocation equalities.
        
        Depends on x, z, x0, p.
        """
        return self._eq


    @property
    def butcher(self):
        """Butecher tableau"""
        return self._butcher


    def combine(self, what):
        """Return a struct_MX combining the specified parts of the collocation scheme.

        @param what is a list of strings with possible values 'x0', 'x', 'z', 'u', 'p', 'eq', 'q'.
        """

        what_set = ['K', 'x', 'Z', 'U', 'u', 'p', 'eq', 'q']
        assert all([w in what_set for w in what])

        return ct.struct_MX([ct.entry(w, expr=getattr(self, w)) for w in what])


    def piecewisePolyX(self, x, K):
        X = self.evalX(x, K)
        M = self._M   

        coeff = []
        for n in range(self._N):
            coeff.append(np.hstack([x[:, n], X[:, n * M : (n + 1) * M]]))

        basis = PolynomialBasis(np.append(0, self._butcher.c))
        return PiecewisePoly(self._t, coeff, basis)


def collocationIntegrator(name, dae, t, order, method='legendre', tdp_fun=None):
    """Make an integrator based on collocation method
    """

    N = order
    scheme = CollocationScheme(dae, t=t, order=order, method=method, tdp_fun=tdp_fun)

    x0 = cs.MX.sym('x0', dae.nx)
    z0 = dae.z

    # Solve the collocation equations w.r.t. (x,K,Z)
    var = scheme.combine(['x', 'K', 'Z'])
    eq = cs.Function('eq', [var, x0, scheme.u, scheme.p], [cs.vertcat(scheme.eq, scheme.x[:, 0] - x0)])
    rf = cs.rootfinder('rf', 'newton', eq)

    # Initial point for the rootfinder
    w0 = ct.struct_MX(var)
    w0['x'] = cs.repmat(x0, 1, scheme.x.shape[1])
    w0['K'] = cs.MX.zeros(scheme.K.shape)
    w0['Z'] = cs.repmat(z0, 1, scheme.Z.shape[1])
    
    sol = var(rf(w0, x0, dae.u, dae.p))
    sol_x = sol['x']
    sol_K = sol['K']
    sol_Z = sol['Z']
    [sol_q, sol_Q, sol_X] = cs.substitute([scheme.q, scheme.Q, scheme.X], 
        [scheme.x, scheme.K, scheme.Z], [sol_x, sol_K, sol_Z])

    # TODO: return correct value for zf!
    return cs.Function(name, 
        [x0, z0, dae.u, dae.p], [sol_x[:, -1], np.repeat(np.nan, dae.nz), sol_q[:, -1], sol_X, sol_Z, sol_Q], 
        ['x0', 'z0', 'u', 'p'], ['xf', 'zf', 'qf', 'X', 'Z', 'Q'])


class CollocationSimulator(object):
    """Simulates DAE system using direct collocation method
    """

    def __init__(self, dae, t, poly_order=5, tdp_fun=None):
        '''Constructor
        '''

        pdq = Pdq(t, poly_order)
        N = len(pdq.collocationPoints)
        scheme = CollocationScheme(dae, pdq, tdp_fun=tdp_fun)

        x0 = cs.MX.sym('x0', dae.nx)
        X = scheme.x
        Z = scheme.z
        z0 = dae.z

        # Solve the collocation equations w.r.t. (X,Z)
        var = scheme.combine(['x', 'z'])
        eq = cs.Function('eq', [var, x0, scheme.u, scheme.p], [cs.vertcat(scheme.eq, scheme.x[:, 0] - x0)])
        rf = cs.rootfinder('rf', 'newton', eq)

        # Initial point for the rootfinder
        w0 = ct.struct_MX(var)
        w0['x'] = cs.repmat(x0, 1, N)
        w0['z'] = cs.repmat(z0, 1, N - 1)
        
        sol = var(rf(w0, x0, scheme.u, scheme.p))
        sol_X = sol['x']
        sol_Z = sol['z']
        [sol_Q] = cs.substitute([scheme.q], [X, Z], [sol_X, sol_Z])

        self._simulate = cs.Function('CollocationSimulator', 
            [x0, z0, scheme.u, scheme.p], [sol_X[:, -1], sol_Z[:, -1], sol_Q[:, -1], sol_X, sol_Z, sol_Q], 
            ['x0', 'z0', 'u', 'p'], ['xf', 'zf', 'qf', 'X', 'Z', 'Q'])

        self._pdq = pdq
        self._dae = dae


    def simulate(self, x0, input, param=None):
        '''Simulate the DAE system
        '''

        if param is None and self._dae.np > 0:
            raise ValueError('DAE system has non-empty parameter vector, but no parameter value was specified.')

        u = cs.horzcat(*[input(t) for t in self._pdq.collocationPoints])
        res = self._simulate(x0=x0, z0=cs.DM.zeros(self._dae.nz), u=u, p=param)

        return ct.SystemTrajectory()
