"""
Collocation method
"""
import numpy as np
import casadi as cs
import casadi_tools as ct


class Pdq(object):
    """Polynomial-based Differential Quadrature (PDQ).

    See "Differential Quadrature and Its Application in Engineering: 
    Engineering Applications, Chang Shu, Springer, 2000, ISBN 978-1-85233-209-9"

    https://link.springer.com/content/pdf/10.1007%2F978-1-4471-0407-0.pdf
    """

    def __init__(self, t, poly_order=5, polynomial_type='cheb'):
        """Constructor

        @param t N+1 points defining N collocation intervals in ascending order.
        @param poly_order order of collocation polynomial.
        @param polynomial_type collocation polynomial type.
        """

        N = len(t) - 1

        # Make collocation points vector and differential matrices list
        D = []
        collocation_points = []
        collocation_groups = []
        k = 0

        for i in range(N):
            D_i, t_i = cheb(poly_order, t[i], t[i + 1])
            
            assert D_i.shape[0] == poly_order + 1 and D_i.shape[1] == poly_order + 1
            assert t_i.shape == (poly_order + 1,)

            # Ensure that the points are from left to right.
            assert np.all(np.diff(t_i) > 0)

            D.append(D_i)
            collocation_points.append(t_i[: -1])
            collocation_groups.append(np.arange(k, k + poly_order + 1))
            k += poly_order

        # Append the last point.
        # TODO: is it needed or not?
        collocation_points.append(t_i[-1])

        # Stack collocation points in one vector
        collocation_points = np.hstack(collocation_points)

        self._D = D
        self._collocationPoints = collocation_points

        # Indices of collocation points belonging to the same interval, including both ends.
        self._collocationGroups = collocation_groups
        self._intervalBounds = t
        self._polyOrder = poly_order


    @property
    def collocationPoints(self):
        """Collocation points"""
        return self._collocationPoints


    @property
    def intervalBounds(self):
        """Interval bounds"""
        return self._intervalBounds


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

        dy = []
        k = 0

        for D in self._D:
            N = D.shape[0] - 1
            dy.append(cs.mtimes(y[:, k : k + N + 1], D[: -1, :].T))
            k += N

        return cs.horzcat(*dy)


    def integral(self, dy):
        """Calculate integral from derivative values
        """

        y0 = cs.MX.zeros(dy.shape[0])
        y = []
        k = 0
        for D in self._D:
            N = D.shape[0] - 1

            # Calculate y the equation 
            # dy = cs.mtimes(y[:, 1 :], D[: -1, 1 :].T)
            #invD = cs.inv(D[: -1, 1 :])
            #y.append(cs.transpose(cs.mtimes(invD, cs.transpose(dy[:, k : k + N]))))
            y.append(cs.transpose(cs.solve(D[: -1, 1 :], cs.transpose(dy[:, k : k + N]))) + cs.repmat(y0, 1, N))
            k += N
            y0 = y[-1][:, -1]

        return cs.horzcat(*y)


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
    """
    if N == 0:
        D = np.array([[0]])
        x = np.array([1])
    else:
        x = np.cos(np.pi * np.arange(N + 1) / N)
        c = np.hstack((2, np.ones(N - 1), 2)) * (-1) ** np.arange(N + 1)
        X = np.tile(x, (N + 1, 1)).T
        dX = X - X.T
        D = np.outer(c, 1 / c) / (dX + np.eye(N + 1)) # off-diagonal entries
        D = D - np.diag(np.sum(D, axis=1))    # diagonal entries

    return np.rot90(2 * D / (tf - t0), 2), (np.flip(x, 0) + 1) / 2 * (tf - t0) + t0


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
        dae_out = dae_map(xdot=pdq.derivative(Xc), x=Xc[:, : -1], z=Zc, u=Uc, p=p, t=tc[: -1], tdp=tdp_val[:, : -1])

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
        self._z = Zc
        self._u = U
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
    def z(self):
        """Algebraic state at collocation points"""
        return self._z


    @property
    def u(self):
        """Control input on control intervals"""
        return self._u


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


def collocationIntegrator(name, dae, pdq):
    """Make an integrator based on collocation method
    """

    N = len(pdq.collocationPoints)
    scheme = CollocationScheme(dae, pdq)

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


def collocation_integrator_index_reduction(name, dae, opts):
    """Make an integrator based on collocation method
    which ensures continuity of z trajectory.

    TODO: this is equivalent to DAE index reduction and then applying the collocation scheme.
    Implement the index reduction routine and support for implicit DAE in collocationScheme() instead.
    """
    x = dae['x']
    z = dae['z'] if 'z' in dae else cs.MX.sym('z', 0)
    xdot = dae['ode']
    alg = dae['alg'] if 'alg' in dae else cs.MX.sym('alg', 0)

    N = opts['collocation_intervals'] if 'collocation_intervals' in opts else 10
    tf = opts['tf'] if 'tf' in opts else 1
    
    D, _ = cheb(N)
    D = D * 2 / tf

    var = ct.struct_symMX([
        ct.entry('X', shape=(x.nnz(), N)),
        ct.entry('Z', shape=(z.nnz(), N + 1))
    ])

    f = cs.Function('f', [x, z], [xdot])
    F = f.map(N, 'serial')

    g = cs.Function('g', [x, z], [alg])
    G = g.map(N, 'serial')

    x0 = x
    z0 = z

    # Time-derivatives obtained by multiplying with the differentiation matrix
    Dx = cs.mtimes(cs.horzcat(var['X'], x0), D.T)

    # System of equations to solve
    eq = [cs.reshape(F(var['X'], var['Z', :, : -1]) - Dx[:, : -1], N * xdot.numel(), 1)]

    # Is there an algebraic equation?
    has_alg = alg.numel() > 0

    if has_alg:
        Dz = cs.mtimes(var['Z'], D.T)

        # Time-derivative of the algebraic equation
        x_dot = cs.MX.sym('x_dot', x.sparsity())
        z_dot = cs.MX.sym('z_dot', z.sparsity())
        g_dot = cs.Function('g_dot', [x, z, x_dot, z_dot], [cs.jtimes(alg, x, x_dot) + cs.jtimes(alg, z, z_dot)])
        G_dot = g_dot.map(N, 'serial')

        eq.append(cs.reshape(G_dot(var['X'], var['Z', :, : -1], Dx[:, : -1], Dz[:, : -1]), N * alg.numel(), 1))
        eq.append(cs.reshape(g(x0, var['Z', :, -1]), alg.numel(), 1))        

    rf = cs.rootfinder('rf', 'newton', cs.Function('eq', [var, x0], [cs.vertcat(*eq)]))

    '''
    The following implementation does not work because of https://github.com/casadi/casadi/issues/2167

    w0 = ct.struct_MX(var)
    w0['X'] = cs.repmat(x0, 1, N)
    w0['Z'] = cs.repmat(z0, 1, N)

    Using a workaround.
    '''
    w0 = cs.vertcat(cs.repmat(x0, N), cs.repmat(z0, N + 1))
    
    sol = var(rf(w0, x0))

    return cs.Function(name, [x0, z0], [sol['X', :, 0], sol['Z', :, 0], sol['X'], sol['Z']], ['x0', 'z0'], ['xf', 'zf', 'X', 'Z'])
