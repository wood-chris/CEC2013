###############################################################################
# Version: 1.1
# Last modified on: 3 April, 2016
# Developers: Michael G. Epitropakis
#      email: m_(DOT)_epitropakis_(AT)_lancaster_(DOT)_ac_(DOT)_uk
###############################################################################

import numpy as np
from . import cfunction as cf


class CF3(cf.CFunction):
    def __init__(self, dim):
        super(CF3, self).__init__(dim, 6)

        # Initialize data for composition
        self._CFunction__sigma_ = np.array([1.0, 1.0, 2.0, 2.0, 2.0, 2.0])
        self._CFunction__bias_ = np.zeros(self._CFunction__nofunc_)
        self._CFunction__weight_ = np.zeros(self._CFunction__nofunc_)
        self._CFunction__lambda_ = np.array([1.0/4.0, 1.0/10.0, 2.0, 1.0, 2.0, 5.0])

        # Lower/Upper Bounds
        self._CFunction__lbound_ = -5.0 * np.ones(dim)
        self._CFunction__ubound_ = 5.0 * np.ones(dim)

        if self.o.shape[1] >= dim:
            self._CFunction__O_ = self.o[:self._CFunction__nofunc_, :dim]
        else:  # randomly initialize
            self._CFunction__O_ = self._CFunction__lbound_ + (self._CFunction__ubound_ - self._CFunction__lbound_) * np.random.rand((self._CFunction__nofunc_, dim))

        # Load M_: Rotation matrices
        if dim in (2, 3, 5, 10, 20):
            fname =  self.function_data_file(3, dim)  #"data/CF3_M_D{}.dat".format(dim)  # + str(dim) + ".dat"
            self._CFunction__load_rotmat(fname)
        else:
            # M_ Identity matrices # TODO: Generate dimension independent rotation matrices
            self._CFunction__M_ = [np.eye(dim)] * self._CFunction__nofunc_

        # Initialize functions of the composition
        self._CFunction__function_ = {0: cf.FEF8F2,
                                      1: cf.FEF8F2,
                                      2: cf.FWeierstrass,
                                      3: cf.FWeierstrass,
                                      4: cf.FGrienwank,
                                      5: cf.FGrienwank}

        # Calculate fmaxi
        self._CFunction__calculate_fmaxi()

    def evaluate(self, x):
        return self._CFunction__evaluate_inner_(x)
