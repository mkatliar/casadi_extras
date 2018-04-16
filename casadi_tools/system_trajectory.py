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
        assert u.shape[1] == Nt
            
        self.input = u
        self._state = x
        self._algState = z
        self._interpolatorX = pdq.interpolator(continuity='both')
        self._interpolatorZ = pdq.interpolator(continuity='right')
        self._pdq = pdq


    def state(self, t):
        return self._interpolatorX(self._state, t)


    def algState(self, t):
        return self._interpolatorZ(self._algState, t)


    @property
    def time(self):
        return self._pdq.intervalBounds

            
