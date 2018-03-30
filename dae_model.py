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

        Dae(x, z, u, p, ode, alg, quad, tdp)
        """

        x = kwargs['x']
        ode = kwargs['ode']

        if not x.is_column():
            raise ValueError('x must be a column vector')

        if not x.is_valid_input():
            raise ValueError('x must be a valid input (purely symbolic)')

        if not ode.is_column():
            raise ValueError('ode must be a column vector')

        if x.numel() != ode.numel():
            raise ValueError('x and ode must have the same number of elements')

        if ('z' in kwargs) != ('alg' in kwargs):
            raise ValueError('if z is specified, then alg must be specified, and vice versa')

        if 'z' in kwargs:
            z = kwargs['z']
            alg = kwargs['alg']

            if not z.is_column():
                raise ValueError('z must be a column vector')

            if not z.is_valid_input():
                raise ValueError('z must be a valid input (purely symbolic)')

            if not alg.is_column():
                raise ValueError('alg must be a column vector')

            if z.numel() != alg.numel():
                raise ValueError('z and alg must have the same number of elements')
        else:
            z = cs.MX.sym('z', 0)
            alg = cs.MX.sym('alg', 0)

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
        self._ode = ode
        self._alg = alg
        self._quad = quad
        self._tdp = tdp


    '''
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
    '''

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
    def ode(self):
        return self._ode


    @property
    def alg(self):
        return self._alg


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


    def createFunction(self, name, in_arg, out_arg):
        return cs.Function(name, [self._get(n) for n in in_arg], [self._get(n) for n in out_arg], in_arg, out_arg)


    def _get(self, name):
        return getattr(self, '_' + name)


def parallel(models):
    '''Connect multiple DAE models in parallel
    '''
    
    d = {}
    for attr in ['x', 'z', 'u', 'p', 'ode', 'alg', 'quad']: # TODO: what do we do with t?
        d[attr] = ct.struct_MX([ct.entry('model_{0}'.format(i), expr=getattr(m, attr)) for i, m in enumerate(models)])

    return Dae(**d)