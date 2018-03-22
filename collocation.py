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

    def __init__(self, D, x):
        """Constructor

        @param D differentiation matrix
        @param x collocation points from left to right.
        """

        assert D.shape[0] == D.shape[1] and D.shape[0] > 0
        N = D.shape[0] - 1

        assert x.shape == (N + 1,)

        # Ensure that the points are from left to right.
        assert np.all(np.diff(x) > 0)

        self._D = D
        self._x = x
        self._N = N


    @property
    def D(self):
        """Differentiation matrix"""
        return self._D


    @property
    def x(self):
        """Collocation points"""
        return self._x


    @property
    def N(self):
        """Degree of interpolating polynomial"""
        return self._N


    @property
    def x0(self):
        """The leftmost collocation point"""
        return self._x[0]


    @property
    def xf(self):
        """The rightmost collocation point"""
        return self._x[-1]


    def intervalLength(self):
        """Distance between the leftmost and the rightmost collocation points"""
        return self._x[-1] - self._x[0]


def polynomialInterpolator(x):
    """Polynomial interpolator with nodes at x"""

    assert x.ndim == 1
    N = x.size - 1
    n = np.atleast_2d(np.arange(N + 1)).T
    X = x ** n

    return lambda u, xx: np.dot(u, np.linalg.solve(X, xx ** n))


def barycentricInterpolator(x):
    """Barycentric interpolator with nodes at x"""

    assert x.ndim == 1
    N = x.size - 1
    n = np.arange(N + 1)

    a = [np.prod(x[j] - x[n[n != j]]) for j in n]

    def p(u, xq):
        r = []

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

    @param N number of collocation intervals.
    @param t distance between the leftmost and the rightmost collocation points.
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

    return Pdq(np.rot90(2 * D / (tf - t0), 2), (np.flip(x, 0) + 1) / 2 * (tf - t0) + t0)


class CollocationScheme(object):
    """Collocation equations on multiple intervals
    for a given DAE model and differentiation matrix.
    """

    def __init__(self, dae, pdq, NT, parallelization='serial'):
        """Constructor

        @param NT number of intervals
        @param pdq Pdq object
        @param dae Dae model
        @param parallelization parallelization of the outer map. Possible set of values is the same as for casadi.Function.map().

        @return Returns a dictionary with the following keys:
        'X' -- state at collocation points
        'Z' -- alg. state at collocation points
        'x0' -- initial state
        'eq' -- the expression eq == 0 defines the collocation equation. eq depends on X, Z, x0, p.
        'Q' -- quadrature values at collocation points depending on x0, X, Z, p.
        """

        N = pdq.N

        #
        # Define variables and functions corresponfing to a single control interval
        # 
        Xc = cs.MX.sym('Xc', dae.nx, N)
        Zc = cs.MX.sym('Zc', dae.nz, N)

        dae_fun = dae.createFunction('dae', ['x', 'z', 'u', 'p'], ['ode', 'alg', 'quad'])
        dae_map = dae_fun.map('dae_map', 'serial', N, [2, 3], [])
        dae_out = dae_map(x=Xc, z=Zc, u=dae.u, p=dae.p)

        x0c = dae.x
        eqc = ct.struct_MX([
            ct.entry('eqc_ode', expr=dae_out['ode'] - cs.mtimes(cs.horzcat(x0c, Xc), pdq.D[1 :, :].T)),
            ct.entry('eqc_alg', expr=dae_out['alg'])
        ])

        # Calculate the quadrature from the equation 
        # Q(var['X'], var['Z'], p) - cs.mtimes(var['Q'], D[: -1, : -1].T)
        Qc = cs.transpose(cs.solve(pdq.D[1 :, 1 :], cs.transpose(dae_out['quad'])))

        #
        # Repeat the scheme NT times
        #
        f_clc = cs.Function('f_clc', [x0c, Xc, Zc, dae.u, dae.p], [eqc.cat, Qc, Qc[:, -1]],
            ['x0', 'X', 'Z', 'u', 'p'], ['eq', 'Q', 'qf'])
        f_clc_map = f_clc.map('f_clc_map', parallelization, NT, [4], [])

        X0 = cs.MX.sym('X0', dae.nx, NT)
        X = cs.MX.sym('X', dae.nx, N * NT)
        Z = cs.MX.sym('Z', dae.nz, N * NT)
        U = cs.MX.sym('U', dae.nu, NT)

        f_clc_out = f_clc_map(x0=X0, X=X, Z=Z, u=U, p=dae.p)

        # Continuity equations
        continuity_eq = X0[:, 1 :] - X[:, np.arange(NT - 1) * N + N - 1]

        # All equations
        eq = ct.struct_MX([
            ct.entry('eq_collocation', expr=f_clc_out['eq']),
            ct.entry('eq_continuity', expr=continuity_eq)
        ])

        self._N = N
        self._NT = NT

        self._eq = eq.cat
        self._x = X
        self._z = Z
        self._u = U
        self._q = f_clc_out['Q']
        self._qf = f_clc_out['qf']
        self._x0 = X0
        self._p = dae.p
        self._t = np.hstack([pdq.x[1 :] + k * pdq.intervalLength() for k in range(NT)])
        self._pdq = pdq


    @property
    def t(self):
        """Collocation points as time vector"""
        return self._t


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
    def qf(self):
        """Quadrature state at the end of each control interval
        
        Depends on x, z, x0, p.
        """
        return self._qf


    @property
    def x0(self):
        """State at the beginning of each control interval"""
        return self._x0
        

    @property
    def xf(self):
        """State at the end of each control interval"""
        return self._x[:, range(self._N - 1, self._NT * self._N, self._N)]


    @property
    def eq(self):
        """Right-hand side of collocation equalities.
        
        Depends on x, z, x0, p.
        """
        return self._eq


    def combine(self, what):
        """Return a struct_MX combining the specified parts of the collocation scheme.

        @param what is a list of strings with possible values 'x0', 'x', 'xf', 'z', 'u', 'p', 'eq', 'q', 'qf'.
        """

        what_set = ['x0', 'x', 'xf', 'z', 'u', 'p', 'eq', 'q', 'qf']
        assert all([w in what_set for w in what])

        return ct.struct_MX([ct.entry(w, expr=getattr(self, w)) for w in what])


    def interpolator(self):
        """Create interpolating function"""

        fi_cl = barycentricInterpolator(self._pdq.x)

        def interp(x0, x, t):
            l = []
            ts = self._pdq.intervalLength()

            for ti in t:
                i = min(max(int(ti // ts), 0), self._NT - 1)  # interval index
                l.append(fi_cl(np.hstack((x0[:, i], x[:, self._N * i : self._N * (i + 1)])), [ti - ts * i]))

            return np.hstack(l)

        return interp


def collocationIntegrator(name, dae, pdq):
    """Make an integrator based on collocation method
    """

    N = pdq.N
    scheme = CollocationScheme(dae, pdq, 1)

    q = dae.createFunction('q', ['x', 'z', 'u', 'p'], ['quad'])
    Q = q.map('Q', 'serial', N, [2], [])

    x0 = scheme.x0
    X = scheme.x
    Z = scheme.z
    z0 = dae.z

    # Solve the collocation equations w.r.t. (X,Z)
    var = scheme.combine(['x', 'z'])
    eq = cs.Function('eq', [var, x0, scheme.u, scheme.p], [scheme.eq])
    rf = cs.rootfinder('rf', 'newton', eq)

    # Initial point for the rootfinder
    w0 = ct.struct_MX(var)
    w0['x'] = cs.repmat(x0, 1, N)
    w0['z'] = cs.repmat(z0, 1, N)
    
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
