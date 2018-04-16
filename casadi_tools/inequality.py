import casadi as cs

from collections import namedtuple


class Inequality(namedtuple('Inequality', ['expr', 'lb', 'ub', 'nominal'])):
    '''Inequality of a form lb <= expr <= ub.
    '''

    def __new__(cls, expr=cs.MX.sym('empty', 0), lb=None, ub=None, nominal=None):
        '''Constructor

        '''

        lb = -cs.DM.inf(expr.shape) if lb is None else cs.DM(lb)
        ub = cs.DM.inf(expr.shape) if ub is None else cs.DM(ub)
        
        assert lb.shape == expr.shape and ub.shape == expr.shape

        if nominal is not None:
            nominal = cs.DM(nominal)
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
