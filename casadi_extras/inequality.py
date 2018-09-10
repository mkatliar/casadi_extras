import casadi as cs

from collections import namedtuple

from .structure3 import struct_MX, entry, DMStruct


def _convertToDmIfNotDmStruct(val):
    return val if isinstance(val, DMStruct) else cs.DM(val)


class Inequality(namedtuple('Inequality', ['expr', 'lb', 'ub', 'nominal'])):
    '''Inequality of a form lb <= expr <= ub.
    '''

    def __new__(cls, expr=cs.MX.sym('empty', 0), lb=None, ub=None, nominal=None):
        '''Constructor

        '''

        lb = -cs.DM.inf(expr.shape) if lb is None else _convertToDmIfNotDmStruct(lb)
        ub = cs.DM.inf(expr.shape) if ub is None else _convertToDmIfNotDmStruct(ub)
        
        assert lb.shape == expr.shape and ub.shape == expr.shape

        if nominal is not None:
            nominal = _convertToDmIfNotDmStruct(nominal)
            assert nominal.shape == expr.shape

        return super().__new__(cls, expr, lb, ub, nominal)


    def numel(self):
        '''Number of elements in the inequality.
        '''
        return self.expr.numel()


class BoundedVariable(Inequality):
    '''A symbolic expression with lower bound, upper bound and a nominal value.
    '''

    def __new__(cls, name='var', lb=None, ub=None, nominal=cs.DM.zeros(0)):
        return super().__new__(cls, cs.MX.sym(name, nominal.shape), lb, ub, nominal)


def vertcat(ineq):
    '''Vertical concatenation of inequalities.
    '''

    if isinstance(ineq, dict):
        expr = struct_MX([entry(k, expr=v.expr) for k, v in ineq.items()])
        lb = expr(dict((k, v.lb) for k, v in ineq.items()))
        ub = expr(dict((k, v.ub) for k, v in ineq.items()))
        nominal = expr(dict((k, v.nominal) for k, v in ineq.items()))
    else:
        expr=cs.vertcat(*[a.expr for a in ineq])
        lb=cs.vertcat(*[a.lb for a in ineq])
        ub=cs.vertcat(*[a.ub for a in ineq])
        nominal=cs.vertcat(*[a.nominal for a in ineq])

    return Inequality(expr=expr, lb=lb, ub=ub, nominal=nominal)