"""
Collocation method
"""
import numpy as np
import casadi as cs
import casadi_tools as ct


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
