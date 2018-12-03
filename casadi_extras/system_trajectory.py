##
## Copyright (c) 2018 Mikhail Katliar.
## 
## This file is part of CasADi Extras 
## (see https://github.com/mkatliar/casadi_extras).
## 
## This program is free software: you can redistribute it and/or modify
## it under the terms of the GNU Lesser General Public License as published by
## the Free Software Foundation, either version 3 of the License, or
## (at your option) any later version.
## 
## This program is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU Lesser General Public License for more details.
## 
## You should have received a copy of the GNU Lesser General Public License
## along with this program. If not, see <http://www.gnu.org/licenses/>.
##
import numpy as np


class SystemTrajectory:
    """
    Contains state and alg state trajectories as continuous-time functions
    plus control input as discrete-time function.
    """

    def __init__(self, x, z, u, pdq):
        """Constructor
        """

        Nt = len(pdq.intervalBounds) - 1
        n_collocation = len(pdq.collocationPoints)

        x = np.atleast_2d(x)
        z = np.atleast_2d(z)
        u = np.atleast_2d(u)
        
        assert x.shape[1] == n_collocation
        assert z.shape[1] == n_collocation - 1
        assert u.shape[1] == n_collocation - 1
            
        self._input = u
        self._state = x
        self._algState = z
        self._interpolatorX = pdq.interpolator(continuity='both')
        self._interpolatorZ = pdq.interpolator(continuity='right')
        self._interpolatorU = pdq.interpolator(continuity='right')
        self._pdq = pdq


    def state(self, t):
        return self._interpolatorX(self._state, t)


    def algState(self, t):
        return self._interpolatorZ(self._algState, t)


    def input(self, t):
        return self._interpolatorU(self._input, t)


    @property
    def time(self):
        return self._pdq.intervalBounds

