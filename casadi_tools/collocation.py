"""
Collocation method
"""
import numpy as np
import casadi as cs
import casadi_tools as ct


def collocationPoints(order, scheme):
    '''Obtain collocation points of specific order and scheme.
    '''
    if scheme == 'chebyshev':
        return np.array([0]) if order == 0 else (1 - np.cos(np.pi * np.arange(order + 1) / order)) / 2
    else:
        return np.append(0, cs.collocation_points(order, scheme))


class PolynomialBasis(object):
    '''Polynomial basis.
    '''
    
    def __init__(self, tau):
        '''Make polynomial basis at the points tau.
        '''

        n = len(tau)
        p = []

        for j in range(n):
            # Construct Lagrange polynomials to get the polynomial basis at the collocation point
            pp = np.poly1d([1])
            for r in range(n):
                if r != j:
                    pp *= np.poly1d([1, -tau[r]]) / (tau[j] - tau[r])

            p.append(pp)

        # Calculate lambdas for barycentric representation
        lam = 1 / np.array([np.prod([tau[k] - tau[np.arange(n) != k]]) for k in range(n)])

        # Construct differentiation matrix.
        #
        # The naive method, which has big error for large n, looks like this:
        # D = np.atleast_2d(np.hstack([np.atleast_2d(np.polyder(pp)(tau)).T for pp in p]))
        #
        # We use a better method from here: http://richard.baltensp.home.hefr.ch/Publications/3.pdf
        
        D = np.zeros((n, n))
        for j in range(n):
            for k in range(n):
                if j != k:
                    D[j, k] = lam[k] / lam[j] / (tau[j] - tau[k])
                else:
                    D[j, k] = -np.sum([lam[i] / lam[j] / (tau[j] - tau[i]) for i in range(n) if i != j])

        self.poly = p
        self.D = D
        self.tau = tau

    
    @property
    def numPoints(self):
        return len(self.tau)


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

        # Transform collocation groups depending on the continuity option.
        groups = []

        for g in self._collocationGroups:
            if continuity == 'both':
                groups.append(g)
            elif continuity == 'left':
                groups.append(g[1 : ])
            elif continuity == 'right':
                groups.append(g[: -1])
            else:
                raise ValueError('Invalid "continuity" value {0} in Pdq.interpolator()'.format(continuity))

        fi_cl = [barycentricInterpolator(self._collocationPoints[g]) for g in groups]

        def interp(x, t):
            expected_x_cols = self._collocationPoints.size if continuity == 'both' else self._collocationPoints.size - 1
            if x.shape[1] != expected_x_cols:
                raise ValueError('Invalid number of columns in interpolation point matrix')

            l = []
            
            for ti in np.atleast_1d(t):
                i = np.clip(np.searchsorted(self._intervalBounds, ti, 'right') - 1, 0, len(self._intervalBounds) - 2)  # interval index
                l.append(fi_cl[i](x[:, groups[i]], ti))

            return np.hstack(l)

        return interp


def polynomialInterpolator(x):
    """Polynomial interpolator with nodes at x"""

    assert x.ndim == 1
    N = x.size - 1
    n = np.atleast_2d(np.arange(N + 1)).T
    X = x ** n

    return lambda u, xx: np.dot(u, np.linalg.solve(X, xx ** n))


def barycentricInterpolator(x):
    """Barycentric interpolator with nodes at x"""

    assert np.ndim(x) == 1
    N = np.size(x) - 1
    n = np.arange(N + 1)
    x = np.atleast_1d(x)    # Convert to numpy type s.t. the indexing below works

    a = [np.prod(x[j] - x[n[n != j]]) for j in n]

    def p(u, xq):
        r = []

        # Convert scalar argument to a vector
        xq = np.atleast_1d(xq)
        
        # Converting to np.array is a workaround for https://github.com/casadi/casadi/issues/2221
        u = np.array(u)

        for xi in xq:
            # Check if xi is in x
            ind = xi == x

            if np.any(ind):
                # At a node
                assert np.sum(ind) == 1
                y = u[:, ind]
            else:
                # Between nodes
                y = np.dot(u, np.atleast_2d(1 / (a * (xi - x))).T) / np.sum(1 / (a * (xi - x)))

            r.append(y)

        return np.hstack(r)

    return p


def cheb(N, t0, tf):
    """Chebyshev grid and differentiation matrix

    The code is based on cheb.m function from the book
    "Spectral Methods in MATLAB" by Lloyd N. Trefethen 
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.473.7647&rep=rep1&type=pdf

    @param N order of collocation polynomial.
    @param t0 left end of the collocation interval
    @param t0 right end of the collocation interval

    @return a tuple (D, t) where D is a (N+1)-by-(N+1) differentiation matrix 
    and t is a vector of points of length N+1 such that t[0] == t0 and t[-1] == tf.

    TODO: deprecate
    """
    
    tau = collocationPoints(N, 'chebyshev')
    basis = PolynomialBasis(tau)
    return basis.D / (tf - t0), tau * (tf - t0) + t0


class CollocationScheme(object):
    """Collocation equations on multiple intervals
    for a given DAE model and differentiation matrix.
    """

    def __init__(self, dae, pdq, parallelization='serial', tdp_fun=None, expand=True, repeat_param=False):
        """Constructor

        @param pdq Pdq object
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

        N = len(pdq.collocationPoints)
        NT = len(pdq.intervalBounds) - 1

        #
        # Define variables and functions corresponfing to all control intervals
        # 
        Xc = cs.MX.sym('Xc', dae.nx, N)
        Zc = cs.MX.sym('Zc', dae.nz, N - 1)
        U = cs.MX.sym('U', dae.nu, NT)
        Uc = cs.horzcat(*[cs.repmat(U[:, k], 1, pdq.polyOrder) for k in range(NT)])

        # Points at which the derivatives are calculated
        tc = pdq.collocationPoints

        # Values of the time-dependent parameter
        if tdp_fun is not None:
            tdp_val = cs.horzcat(*[tdp_fun(t) for t in tc])
        else:
            assert dae.ntdp == 0
            tdp_val = np.zeros((0, N))

        # DAE function
        dae_fun = dae.createFunction('dae', ['xdot', 'x', 'z', 'u', 'p', 't', 'tdp'], ['dae', 'quad'])
        if expand:
            dae_fun = dae_fun.expand()  # expand() for speed

        if repeat_param:
            reduce_in = []
            p = cs.MX.sym('P', dae.np, N - 1)
        else:
            reduce_in = [4]
            p = cs.MX.sym('P', dae.np)

        dae_map = dae_fun.map('dae_map', parallelization, N - 1, reduce_in, [])

        xdot = pdq.derivative(Xc)
        dae_out = dae_map(xdot=xdot, x=Xc[:, : -1], z=Zc, u=Uc, p=p, t=tc[: -1], tdp=tdp_val[:, : -1])

        eqc = [
            cs.vec(dae_out['dae']),
            cs.vec(cs.diff(p, 1, 1))
        ]

        # Calculate the quadrature
        Qc = pdq.integral(dae_out['quad'])

        self._N = N
        self._NT = NT

        self._eq = cs.vertcat(*eqc)
        self._x = Xc
        self._xdot = xdot
        self._z = Zc
        self._u = U
        self._uc = Uc
        self._q = Qc
        self._x0 = Xc[:, range(0, N - 1, pdq.polyOrder)]
        self._p = p
        self._t = tc
        self._pdq = pdq
        self._tdp = tdp_val


    @property
    def pdq(self):
        """PDQ used by the collocation scheme"""
        return self._pdq


    @property
    def t(self):
        """Collocation points as time vector"""
        return self._t


    @property
    def numTotalCollocationPoints(self):
        """Total number of collocation points"""
        return self._t.size


    @property
    def x(self):
        """State at collocation points"""
        return self._x


    @property
    def xdot(self):
        """State derivative at collocation points"""
        return self._xdot


    @property
    def z(self):
        """Algebraic state at collocation points"""
        return self._z


    @property
    def u(self):
        """Control input on control intervals"""
        return self._u


    @property
    def uc(self):
        """Control input at collocation points"""
        return self._uc


    @property
    def p(self):
        """DAE model parameters"""
        return self._p


    @property
    def q(self):
        """Quadrature state at collocation points"""
        return self._q


    @property
    def x0(self):
        """State at the beginning of each control interval
        
        TODO: deprecate?
        """
        return self._x0
        

    @property
    def eq(self):
        """Right-hand side of collocation equalities.
        
        Depends on x, z, x0, p.
        """
        return self._eq


    def combine(self, what):
        """Return a struct_MX combining the specified parts of the collocation scheme.

        @param what is a list of strings with possible values 'x0', 'x', 'z', 'u', 'p', 'eq', 'q'.
        """

        what_set = ['x0', 'x', 'z', 'u', 'p', 'eq', 'q']
        assert all([w in what_set for w in what])

        return ct.struct_MX([ct.entry(w, expr=getattr(self, w)) for w in what])


def collocationIntegrator(name, dae, pdq, tdp_fun=None):
    """Make an integrator based on collocation method
    """

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
    
    sol = var(rf(w0, x0, dae.u, dae.p))
    sol_X = sol['x']
    sol_Z = sol['z']
    [sol_Q] = cs.substitute([scheme.q], [X, Z], [sol_X, sol_Z])

    return cs.Function(name, [x0, z0, dae.u, dae.p], [sol_X[:, -1], sol_Z[:, -1], sol_Q[:, -1], sol_X, sol_Z, sol_Q], 
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
