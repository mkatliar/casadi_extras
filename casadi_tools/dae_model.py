"""
Support for DAE formulation
"""

import casadi as cs
import casadi_tools as ct


class Dae(object):
    """DAE formulation
    """

    def __init__(self, **kwargs):
        """Constructor

        Dae(x, z, u, p, t, quad, tdp)
        """

        x = kwargs['x']
        
        if not x.is_column():
            raise ValueError('x must be a column vector')

        if not x.is_valid_input():
            raise ValueError('x must be a valid input (purely symbolic)')

        if 'z' in kwargs:
            z = kwargs['z']

            if not z.is_column():
                raise ValueError('z must be a column vector')

            if not z.is_valid_input():
                raise ValueError('z must be a valid input (purely symbolic)')
        else:
            z = cs.MX.sym('z', 0)

        if 'u' in kwargs:
            u = kwargs['u']

            if not u.is_column():
                raise ValueError('u must be a column vector')

            if not u.is_valid_input():
                raise ValueError('u must be a valid input (purely symbolic)')
        else:
            u = cs.MX.sym('u', 0)

        if 'p' in kwargs:
            p = kwargs['p']

            if not p.is_valid_input():
                raise ValueError('p must be a valid input (purely symbolic)')

        else:
            p = cs.MX.sym('p', 0)

        if 't' in kwargs:
            t = kwargs['t']

            if not t.is_valid_input():
                raise ValueError('t must be a valid input (purely symbolic)')

            if not t.is_scalar():
                raise ValueError('t must be scalar')

        else:
            t = cs.MX.sym('t')

        if 'quad' in kwargs:
            quad = kwargs['quad']
        else:
            quad = cs.MX.sym('quad', 0)

        if 'tdp' in kwargs:
            tdp = kwargs['tdp']

            if not tdp.is_valid_input():
                raise ValueError('tdp must be a valid input (purely symbolic)')

        else:
            tdp = cs.MX.sym('tdp', 0)

        self._x = x
        self._z = z
        self._u = u
        self._p = p
        self._t = t
        self._quad = quad
        self._tdp = tdp


    @property
    def x(self):
        return self._x


    @property
    def z(self):
        return self._z


    @property
    def u(self):
        return self._u


    @property
    def p(self):
        return self._p


    @property
    def t(self):
        return self._t


    @property
    def quad(self):
        return self._quad


    @property
    def tdp(self):
        return self._tdp


    @property
    def nx(self):
        return self._x.numel()


    @property
    def nz(self):
        return self._z.numel()


    @property
    def nu(self):
        return self._u.numel()


    @property
    def np(self):
        return self._p.numel()


    @property
    def ntdp(self):
        return self._tdp.numel()


    @property
    def nq(self):
        return self._quad.numel()


    def createFunction(self, name, in_arg, out_arg):
        return cs.Function(name, [self._get(n) for n in in_arg], [self._get(n) for n in out_arg], in_arg, out_arg)


    def _get(self, name):
        return getattr(self, '_' + name)


class SemiExplicitDae(Dae):
    '''Semi-explicit DAE
    '''

    def __init__(self, **kwargs):
        """Constructor

        SemiExplicitDae(x, z, u, p, t, ode, alg, quad, tdp)
        """

        super().__init__(**kwargs)

        ode = kwargs['ode']
        if not ode.is_column():
            raise ValueError('ode must be a column vector')

        if self.x.numel() != ode.numel():
            raise ValueError('x and ode must have the same number of elements')

        if 'alg' in kwargs:
            alg = kwargs['alg']

            if not alg.is_column():
                raise ValueError('alg must be a column vector')

            if self.z.numel() != alg.numel():
                raise ValueError('z and alg must have the same number of elements')
        else:
            alg = cs.MX.sym('alg', 0)

        self._ode = ode
        self._alg = alg


    @property
    def ode(self):
        return self._ode


    @property
    def alg(self):
        return self._alg


    def makeImplicit(self):
        '''Convert to implicit DAE
        '''
        xdot = cs.MX.sym('xdot', self.nx)

        return ImplicitDae(xdot=xdot, x=self.x, z=self.z, u=self.u, p=self.p, t=self.t,
            dae=cs.vertcat(xdot - self.ode, self.alg), quad=self.quad, tdp=self.tdp)


class ImplicitDae(Dae):
    '''Implicit DAE
    '''

    def __init__(self, **kwargs):
        """Constructor

        ImplicitDae(xdot, x, z, u, p, t, dae, quad, tdp)
        """

        super().__init__(**kwargs)

        xdot = kwargs['xdot']
        if not xdot.is_column():
            raise ValueError('xdot must be a column vector')

        if self.x.numel() != xdot.numel():
            raise ValueError('x and xdot must have the same number of elements')

        dae = kwargs['dae']

        if not dae.is_column():
            raise ValueError('dae must be a column vector')

        if self.z.numel() + xdot.numel() != dae.numel():
            raise ValueError('The size in dae must be equal to size of x + size of z')

        self._xdot = xdot
        self._dae = dae


    @property
    def xdot(self):
        return self._xdot


    @property
    def dae(self):
        return self._dae


    def makeImplicit(self):
        '''Convert to implicit DAE
        '''
        return self


def parallel(models):
    '''Connect multiple DAE models in parallel
    '''
    
    d = {}
    for attr in ['x', 'z', 'u', 'p', 'ode', 'alg', 'quad']: # TODO: what do we do with t?
        d[attr] = ct.struct_MX([ct.entry('model_{0}'.format(i), expr=getattr(m, attr)) for i, m in enumerate(models)])

    return Dae(**d)