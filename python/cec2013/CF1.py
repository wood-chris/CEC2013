###############################################################################
# Version: 1.1
# Last modified on: 3 April, 2016
# Developers: Michael G. Epitropakis
#      email: m_(DOT)_epitropakis_(AT)_lancaster_(DOT)_ac_(DOT)_uk
###############################################################################

import numpy as np
import cfunction as cf


class CF1(cf.CFunction):

    def __init__(self, dim):
        super(CF1, self).__init__(dim, 6)

        # Initialize data for composition
        self._CFunction__sigma_ = np.ones(self._CFunction__nofunc_)
        self._CFunction__bias_ = np.zeros(self._CFunction__nofunc_)
        self._CFunction__weight_ = np.zeros(self._CFunction__nofunc_)
        self._CFunction__lambda_ = np.array([1.0, 1.0, 8.0, 8.0, 1.0/5.0, 1.0/5.0])

        # Lower/Upper Bounds
        self._CFunction__lbound_ = -5.0 * np.ones(dim)
        self._CFunction__ubound_ = 5.0 * np.ones(dim)

        # Load optima
        if self.o.shape[1] >= dim:
            self._CFunction__O_ = self.o[:self._CFunction__nofunc_, :dim]
        else:  # randomly initialize
            self._CFunction__O_ = self._CFunction__lbound_ + (self._CFunction__ubound_ - self._CFunction__lbound_) * np.random.rand((self._CFunction__nofunc_, dim))

        # M_: Identity matrices
        self._CFunction__M_ = [np.eye(dim)] * self._CFunction__nofunc_

        # Initialize functions of the composition
        self._CFunction__function_ = {0: cf.FGrienwank,
                                      1: cf.FGrienwank,
                                      2: cf.FWeierstrass,
                                      3: cf.FWeierstrass,
                                      4: cf.FSphere,
                                      5: cf.FSphere}

        # Calculate fmaxi
        self._CFunction__calculate_fmaxi()

    def evaluate(self, x):
        return self._CFunction__evaluate_inner_(x)
